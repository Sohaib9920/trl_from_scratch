import torch.nn as nn
from transformers import AutoModelForCausalLM
import torch

class ValueModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype = getattr(torch, config["torch_dtype"])
        self.transformer = AutoModelForCausalLM.from_pretrained(config['lm_name'], 
                                                                torch_dtype=dtype,
                                                                device_map=config["device_map"],
                                                                output_hidden_states=True)
        self.v_head = nn.Linear(self.transformer.config.hidden_size, 1, dtype=dtype)
    
    def forward(self, *args, **kwargs):
        out = self.transformer(*args, **kwargs)
        value = self.v_head(out.hidden_states[-1]).squeeze(-1)
        return out.logits, out.past_key_values, value