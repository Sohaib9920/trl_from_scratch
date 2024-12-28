import torch
import numpy as np

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [d[k] for d in stats_dicts]
        results[k] = torch.stack(stats_list)
    return results

def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened

def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, dict):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict