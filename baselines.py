import math

import torch
import torch.nn.functional as F

from lizard_attention import window_mask


def fla_gla(fla_fn, fq, fk, v, gamma):
    v1 = torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)
    q, k, v1 = (t.transpose(1, 2).contiguous() for t in (fq, fk, v1))
    g = gamma.log()[:, :, None].expand(-1, -1, fq.shape[1]).contiguous()
    o = fla_fn(q, k, v1, g=g, scale=1.0)[0].transpose(1, 2)
    return o[..., :-1] / o[..., -1:]


def sink(meta, heads):
    return torch.logsumexp(meta, 0).expand(heads).contiguous()


def additive_mask(length, window, dtype):
    mask = window_mask(length, window)
    zeros = torch.zeros(length, length, dtype=dtype)
    return zeros.masked_fill(~mask, float("-inf"))


def pytorch_sdpa_awa(q, k, v, meta, window):
    b, h, n, d = q.shape
    zero = torch.zeros(b, h, 1, d, dtype=q.dtype, device=q.device)
    sink_column = torch.full((n, 1), sink(meta, 1).item(), dtype=q.dtype)
    bias = torch.cat([additive_mask(n, window, q.dtype), sink_column], -1)
    return F.scaled_dot_product_attention(
        q, torch.cat([k, zero], 2), torch.cat([v, zero], 2),
        attn_mask=bias.to(q.device))


def hf_gpt_oss_awa(q, k, v, meta, window):
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        eager_attention_forward)
    module = torch.nn.Module()
    module.sinks = sink(meta, q.shape[1])
    module.num_key_value_groups = 1
    mask = additive_mask(q.shape[2], window, q.dtype).to(q.device)
    out, _ = eager_attention_forward(module, q, k, v, mask[None, None],
                                     scaling=q.shape[-1] ** -0.5)
    return out.transpose(1, 2)


def openai_gpt_oss_awa(q, k, v, meta, window):
    from gpt_oss.torch.model import sdpa
    _, h, n, d = q.shape
    q_, k_, v_ = (t[0].transpose(0, 1) for t in (q, k, v))
    out = sdpa(q_[:, :, None], k_, v_, sink(meta, h), 1 / math.sqrt(d),
               window)
    return out.view(n, h, d).transpose(0, 1)[None]


def fla_awa(q, k, v, meta, window, kernel=False):
    from fla.ops.attn import naive_parallel_attn, parallel_attn
    fn = parallel_attn if kernel else naive_parallel_attn
    out = fn(*(t.transpose(1, 2).contiguous() for t in (q, k, v)),
             window_size=window, sink_bias=sink(meta, q.shape[1]))
    return (out[0] if isinstance(out, tuple) else out).transpose(1, 2)
