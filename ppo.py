import torch
import time
import numpy as np
from torch.optim import Adam
from utils import stack_dicts, stats_to_np, flatten_dict, whiten


class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """
    
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True, 
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 16,
        "ppo_epochs": 4,    
    } 
    
    def __init__(self, model, ref_model, **ppo_params):
        """
        Initialize PPOTrainer.
        
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
                
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        
        self.ref_model = ref_model
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])
        
        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])


    def step(self, query, response, scores):
        """
        Run a PPO optimisation step.
        
        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]
            
        returns:
            train_stats (dict): a summary of the training statistics
        """

        timing = dict()
        t0 = time.time()
        
        # rollout phase finding logprobs (for both policy and ref_policy) and rewards (for policy only) of
        # `batch_size` number of tragectories of states
        t = time.time()
        with torch.no_grad():
            logprobs, ref_logprobs, values = self.batched_forward_pass(query, response)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t 

        # Finding advantages and returns to teach policy (actor) and value network (critic) respectively
        t = time.time() 
        advantages, returns = self.compute_adv_returns(rewards, values)
        timing['time/ppo/compute_advantages_returns'] = time.time()-t
        
        # train policy and value head on rollout data for `ppo_epochs` number of epochs by finding new logprobs and predicted v values
        t = time.time() 
        train_stats = self.train_policy(query, response, logprobs, values, advantages, returns)
        timing['time/ppo/optimize_step'] = time.time()-t
        
        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        
        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats


    def train_policy(self, query, response, logprobs, values, advantages, returns):
        bs = self.ppo_params['batch_size']
        sbs = self.ppo_params['step_batch_size']
        b_inds = list(range(bs))
        model_input = torch.cat([query, response], dim=1)
        all_stats = []

        for _ in range(self.ppo_params['ppo_epochs']):
            np.random.shuffle(b_inds)

            for start in range(0, bs, sbs):
                end = start + sbs
                mb_inds = b_inds[start:end]
                mb_query = query[mb_inds]
                mb_response = response[mb_inds]
                mb_model_input = model_input[mb_inds]
                mb_logprobs = logprobs[mb_inds]
                mb_values = values[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_returns = returns[mb_inds]

                train_stats = self.train_minibatch(mb_query, mb_response, mb_model_input,
                                                   mb_logprobs, mb_values, mb_advantages, mb_returns)

                all_stats.append(train_stats)
        
        train_stats = stack_dicts(all_stats)
        return train_stats


    def batched_forward_pass(self, query, response):
        gen_len = response.shape[1]
        fbs = self.ppo_params['forward_batch_size']
        model_input = torch.cat((query, response), axis=1)
        logprobs = []
        ref_logprobs = []
        values = []
        
        for i in range(int(self.ppo_params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            logits, _, v = self.model(m_input)
            ref_logits, _, _ = self.ref_model(m_input)
            lp = logits[:, :-1, :].log_softmax(-1).gather(-1, m_input[:, 1:, None]).squeeze(-1)[:, -gen_len:]
            ref_lp = ref_logits[:, :-1, :].log_softmax(-1).gather(-1, m_input[:, 1:, None]).squeeze(-1)[:, -gen_len:]
            values.append(v[:, -gen_len-1:-1])
            logprobs.append(lp)
            ref_logprobs.append(ref_lp)
        
        logprobs = torch.cat(logprobs)
        ref_logprobs = torch.cat(ref_logprobs)
        values = torch.cat(values)

        return logprobs, ref_logprobs, values
    

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone()
        rewards[:, -1] += scores
        return rewards, non_score_reward, self.kl_ctl.value


    def compute_adv_returns(self, rewards, values):
        lastgaelam = 0.0
        advantages_reversed = []

        for t in reversed(range(rewards.shape[1])):
            nextvalues = rewards[:, t + 1] if t < rewards.shape[1] - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params["gamma"] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
            
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values

        return advantages, returns
    

    def train_minibatch(self, query, response, model_input,
                        logprobs, values, advantages, returns):
        
        """Train one PPO minibatch"""
        loss, train_stats = self.loss(query, response, model_input, logprobs, values, advantages, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats


    def loss(self, query, response, model_input, logprobs, values, advantages, returns):
        gen_len = response.shape[1]
        new_logits, _, vpred = self.model(model_input)
        new_logprobs = new_logits[:, :-1, :].log_softmax(-1).gather(-1, model_input[:, 1:, None]).squeeze(-1)
        new_logprobs, vpred = new_logprobs[:, -gen_len:], vpred[:,-gen_len-1:-1]
        
        logratio = new_logprobs - logprobs
        ratio = logratio.exp()

        # normalize advantages
        advantages = whiten(advantages)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.ppo_params['cliprange'], 1.0 + self.ppo_params['cliprange'])
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        pg_clipfrac = torch.gt(pg_loss2, pg_loss1).double().mean()

        # value loss 
        vpredclipped = torch.clamp(vpred, values - self.ppo_params["cliprange_value"], values + self.ppo_params["cliprange_value"])
        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.max(vf_losses1, vf_losses2).mean()
        vf_clipfrac =  torch.gt(vf_losses2, vf_losses1).double().mean()

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        with torch.no_grad():
            lp = new_logits.log_softmax(dim=-1)
            entropy = (-lp.exp()*lp).sum(-1).mean()
            approxkl = .5 * torch.mean((new_logprobs - logprobs)**2)
            policykl = torch.mean(new_logprobs - logprobs)
            return_mean, return_var = torch.mean(returns), torch.var(returns)
            value_mean, value_var = torch.mean(values), torch.var(values)

        vpred, new_logprobs, ratio = vpred.detach(), new_logprobs.detach(), ratio.detach()
        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(entropy=entropy, approxkl=approxkl, policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                        clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )

        return loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        # sum because we are finding metrics for trajectories (log(p(T))=Sum[log p(tokens)])
        # expectation comes from average over trajectories
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats
    
