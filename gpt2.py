import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Model
import torch
import torch.nn.functional as F


class GPT2HeadWithValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
       
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
        
        return outputs


def respond_to_batch(model, queries, txt_len=20):
    """Sample text from language model."""
    input_ids = queries
    past = None
    responses = []
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids, past_key_values=past)
        next_token_logits = outputs[0][:, -1, :]
        past = outputs[1]
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        responses.append(next_token.clone())
        input_ids = next_token
        
    return torch.cat(responses, dim=-1)