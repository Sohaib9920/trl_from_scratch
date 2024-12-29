import torch
import wandb
import time
from tqdm import tqdm
import numpy as np
tqdm.pandas()
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from gpt2 import GPT2HeadWithValueModel, respond_to_batch
from ppo import PPOTrainer
import random
import os

config = {
    "lm_name": "lvwerra/gpt2-imdb",
    "ref_lm_name": "lvwerra/gpt2-imdb",
    "cls_model_name": "lvwerra/distilbert-imdb",
    "tk_name": "gpt2",
    "steps": 25600,
    "batch_size": 256,
    "forward_batch_size": 16,
    "step_batch_size": 1,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(project='gpt2-sentiment', config=config)

ds = load_dataset('imdb', split='train')
ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
df = ds.to_pandas()
# make sure the comments are long enough
df = df.loc[df['review'].str.len() > 500]
# make sure comments are not too long
df['review'] = df['review'].apply(lambda x: x[:1000])

sentiment_model = AutoModelForSequenceClassification.from_pretrained(config["cls_model_name"])
sentiment_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
gpt2_tokenizer = AutoTokenizer.from_pretrained(config['tk_name'])

wandb.watch(gpt2_model, log='all')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)
sentiment_model.to(device)
gpt2_model_ref.to(device)

df['tokens'] = df['review'].progress_apply(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])
df['query'] = df['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))

# The training loop consists of the following steps:
# 1. Get a batch of queries
# 2. Get the query responses from the policy
# 3. Join query and responses and tokenize for BERT analysis
# 4. Get sentiments for query/responses from BERT
# 5. Optimize policy with PPO using the (query, response, reward) triplet
# 6. Log all the training statistics

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **config)

for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):

    # init
    fbs = config["forward_batch_size"]
    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()

    ########## sample `batch_size` number of trajectories #############
    df_batch = df.sample(config["batch_size"])

    # get queries (initial states)
    game_data['query'] = df_batch['query'].tolist()
    query_tensors = torch.stack(df_batch['tokens'].tolist())

    # sample their responses (next states)
    t = time.time()
    total_length = config['txt_in_len']+config['txt_out_len']
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        with torch.no_grad():
            response  = respond_to_batch(gpt2_model, query_tensors[i*fbs:(i+1)*fbs],
                                        txt_len=config['txt_out_len'])
            response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    game_data['response'] = gpt2_tokenizer.batch_decode(response_tensors)
    timing['time/get_response'] = time.time()-t

    ########## Reward the tracjectories ##############
    t = time.time()
    texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
    sentiment_inputs, attention_masks = sentiment_tokenizer(texts, padding=True, return_tensors="pt").to(device).values()    
    timing['time/build_input_sentiment'] = time.time()-t

    t = time.time()
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        with torch.no_grad():
            res = sentiment_model.forward(sentiment_inputs[i*fbs:(i+1)*fbs],
                                        attention_masks[i*fbs:(i+1)*fbs])[0][:, 1]
            rewards.append(res)
    rewards = torch.cat(rewards)
    timing['time/get_sentiment_preds'] = time.time()-t

    ######## PPO optimization ###############
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t

    ########### Log everything ###############
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
    logs.update({'game_log':wandb.Table(
        columns=['query', 'response', 'reward'],
        rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)
