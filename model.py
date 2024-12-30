import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch

class ValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(config['lm_name'], 
                                                                torch_dtype=getattr(torch, config["torch_dtype"]),
                                                                device_map=config["device_map"])
        self.v_head = nn.Linear(self.transformer.config.hidden_size, 1)
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)
        value = self.v_head(out.hidden_states).squeeze(-1)
        return out.logits, out.past_key_values, value