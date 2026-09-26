import math

import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import repeat_kv

from lizard_attention import LizardAttention


def rel_err(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def hedgehog_loop(x, weight):
    xw = x @ weight.T
    pos, neg = torch.exp(xw), torch.exp(-xw)
    return torch.cat([pos / pos.sum(-1, keepdim=True),
                      neg / neg.sum(-1, keepdim=True)], dim=-1)


def gla_loop(fq, fk, v, gamma):
    out = torch.zeros_like(v)
    for i in range(fq.shape[2]):
        num, den = 0, 0
        for t in range(i + 1):
            decay = gamma[:, t + 1:i + 1].prod(-1)[:, None, None]
            w = decay * (fq[:, :, i] * fk[:, :, t]).sum(-1, keepdim=True)
            num, den = num + w * v[:, :, t], den + w
        out[:, :, i] = num / den
    return out


def gla_recurrent(fq, fk, v, gamma):
    b, h, n, f = fq.shape
    S = torch.zeros(b, h, f, v.shape[-1], dtype=fq.dtype)
    z = torch.zeros(b, h, f, dtype=fq.dtype)
    out = []
    for i in range(n):
        g = gamma[:, i, None, None]
        S = g[..., None] * S + fk[:, :, i, :, None] * v[:, :, i, None, :]
        z = g * z + fk[:, :, i]
        num = torch.einsum("bhf,bhfd->bhd", fq[:, :, i], S)
        out.append(num / (fq[:, :, i] * z).sum(-1, keepdim=True))
    return torch.stack(out, dim=2)


def awa_loop(q, k, v, meta, window):
    out = torch.zeros_like(v)
    for i in range(q.shape[2]):
        s = max(0, i - window + 1)
        scores = (q[:, :, i, None] * k[:, :, s:i + 1]).sum(-1)
        e = torch.exp(scores / math.sqrt(q.shape[-1]))
        num = (e[..., None] * v[:, :, s:i + 1]).sum(-2)
        out[:, :, i] = num / (meta.exp().sum() + e.sum(-1, keepdim=True))
    return out


def lizard_loop(layer, x):
    p = {n: t.detach().double() for n, t in layer.named_parameters()}
    x = x.double()
    b, n, _ = x.shape

    def heads(name, count):
        y = (x @ p[name].T).view(b, n, count, -1).transpose(1, 2)
        return repeat_kv(y, layer.heads // count)

    q = heads("q_proj.weight", layer.heads)
    k = heads("k_proj.weight", layer.kv_heads)
    v = heads("v_proj.weight", layer.kv_heads)
    gamma = 1 / (1 + torch.exp(-(x @ p["W_gamma.weight"].T))).squeeze(-1)
    fq = hedgehog_loop(q, p["phi_q.weight"])
    fk = hedgehog_loop(k, p["phi_k.weight"])
    y = gla_loop(fq, fk, v, gamma) + p["alpha_blend"] * awa_loop(
        q, k, v, p["meta_tokens"], layer.window)
    return y.transpose(1, 2).reshape(b, n, -1) @ p["o_proj.weight"].T


def tiny_config(layers=1):
    return LlamaConfig(hidden_size=128, intermediate_size=256,
                       num_hidden_layers=layers, num_attention_heads=4,
                       num_key_value_heads=2, vocab_size=100,
                       attn_implementation="eager")


def random_layer(seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    layer = LizardAttention(tiny_config(), 0, window=16, feature_dim=8)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0, 0.3)
    return layer.to(dtype)


def gla_inputs(length, batch=2, heads=3, features=6, dim=5):
    fq = torch.randn(batch, heads, length, features).softmax(-1).double()
    fk = torch.randn(batch, heads, length, features).softmax(-1).double()
    v = torch.randn(batch, heads, length, dim).double()
    gamma = torch.sigmoid(2 * torch.randn(batch, length)).double()
    return fq, fk, v, gamma


def awa_inputs(length, batch=2, heads=3, dim=8):
    q, k, v = (torch.randn(batch, heads, length, dim).double()
               for _ in range(3))
    return q, k, v, torch.randn(4).double()
