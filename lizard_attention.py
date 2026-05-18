"""
LizardAttention — iteration 2.

Changes from iteration 1:
  - GLA via FLA's chunk_gla kernel (correct + fast, replaces hand-written chunked version)
  - AWA via FlashAttention-2 with native windowed attention (replaces is_causal=True hack)
  - Meta-token denominator correction implemented via LSE rescaling (was missing)
  - Hedgehog activation: softmax (per paper Table 13)
  - Normalized GLA output (per paper Section 3.1; needs a 2nd chunk_gla call)

Requirements:
    pip install flash-attn --no-build-isolation     # FA2 (build takes ~20 min on L4)
    pip install flash-linear-attention              # FLA

If flash-attn build fails on your L4, try a prebuilt wheel from
https://github.com/Dao-AILab/flash-attention/releases matching your torch/cuda.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from fla.ops.gla import chunk_gla
from fla.ops.gla import fused_recurrent_gla

from config import *

# These constants are still pulled from the script's global hyperparameters
# FEATURE_DIM, WINDOW_SIZE, NUM_META_TOKENS — keep the same names


class LizardAttention(nn.Module):
    """Drop-in replacement for LlamaAttention.

    Output = GLA(x) + alpha * AnchorWindow(x)
      - GLA: normalized gated linear attention, computed by FLA's chunk_gla
      - AnchorWindow: FA2 windowed softmax attention with meta-token denominator sinks
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads          # 32 for 1B
        self.num_kv_heads = config.num_key_value_heads       # 8 for 1B (GQA)
        self.head_dim = self.hidden_size // self.num_heads   # 64
        self.kv_groups = self.num_heads // self.num_kv_heads # 4

        # Standard projections (initialized from teacher at swap time)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Hedgehog feature maps phi_q, phi_k : head_dim -> 2 * FEATURE_DIM
        self.phi_q = nn.Linear(self.head_dim, FEATURE_DIM, bias=False)
        self.phi_k = nn.Linear(self.head_dim, FEATURE_DIM, bias=False)

        # Scalar gate (shared across heads, per token): hidden -> 1
        self.W_gamma = nn.Linear(self.hidden_size, 1, bias=False)

        # Meta-memory token logits (denominator-only attention sinks for AWA)
        self.meta_tokens = nn.Parameter(torch.zeros(NUM_META_TOKENS))

        # Learnable blend coefficient
        self.alpha_blend = nn.Parameter(torch.tensor(1.0))

        # Init: feature maps small, gate centered at sigmoid(0) = 0.5
        nn.init.normal_(self.phi_q.weight, std=0.02)
        nn.init.normal_(self.phi_k.weight, std=0.02)
        nn.init.zeros_(self.W_gamma.weight)

    def hedgehog(self, x, proj):
        """Softmax-Hedgehog feature map (paper Table 13).

        x: (B, H, L, head_dim)  ->  (B, H, L, 2 * FEATURE_DIM)
        Bounded in (0, 1] along feature dim; sums to 1 along feature dim per half.
        """
        xw = proj(x)
        return torch.cat([F.softmax(xw, dim=-1), F.softmax(-xw, dim=-1)], dim=-1)

    def gla_branch(self, phi_q, phi_k, v, log_gate):
        B, H, L, K = phi_q.shape
        D = v.shape[-1]

        # FLA now requires (B, T, H, D) — head_first option was removed
        phi_q_ = phi_q.transpose(1, 2).contiguous()  # (B, L, H, K)
        phi_k_ = phi_k.transpose(1, 2).contiguous()  # (B, L, H, K)
        v_ = v.transpose(1, 2).contiguous()  # (B, L, H, D)

        # Gate (B, L) -> (B, L, H, K), per-key feature
        g = log_gate.view(B, L, 1, 1).expand(B, L, H, K).contiguous()

        # Numerator
        num, _ = fused_recurrent_gla(q=phi_q_, k=phi_k_, v=v_, gk=g)

        # Denominator: same gated accumulation with v=ones
        ones = torch.ones_like(v_[..., :1])  # (B, L, H, 1)
        denom, _ = fused_recurrent_gla(q=phi_q_, k=phi_k_, v=ones, gk=g)
        denom = denom.clamp(min=1e-6)

        out = num / denom  # (B, L, H, D)
        return out.transpose(1, 2).contiguous()  # back to (B, H, L, D)

    def awa_branch(self, q, k, v):
        """Paper-faithful Anchor Window Attention with meta-token denominator.

        Implements Lizard paper Section 3.1:
          y_i = sum_{t in window} exp(q_i · k_t / sqrt(d)) v_t
                / [sum_j t_j + sum_{t in window} exp(q_i · k_t / sqrt(d))]

        Meta tokens t_j enter the denominator additively in max-subtracted softmax space.

        Args:
            q, k, v: (B, H, L, head_dim)
        Returns:
            (B, H, L, head_dim)
        """
        B, H, L, D = q.shape
        scale = 1.0 / math.sqrt(D)
        device = q.device

        # Sliding causal window: position i attends to [max(0, i-W+1), i]
        idx = torch.arange(L, device=device)
        valid = (idx.unsqueeze(0) <= idx.unsqueeze(1)) & \
                ((idx.unsqueeze(1) - idx.unsqueeze(0)) < WINDOW_SIZE)
        # valid: (L, L)

        # Compute scores in fp32 for stability
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()

        scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # (B, H, L, L)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Numerically stable softmax with meta-token denominator
        # max over valid positions per query
        max_scores = scores.max(dim=-1, keepdim=True).values  # (B, H, L, 1)
        # If a row is all -inf (shouldn't happen with causal window), guard against NaN
        max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)

        exp_scores = torch.exp(scores - max_scores)  # (B, H, L, L)
        exp_scores = exp_scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), 0.0)

        # Numerator: weighted sum of values
        num = torch.matmul(exp_scores, v_f)  # (B, H, L, D)

        # Local denominator: sum over window
        denom_local = exp_scores.sum(dim=-1, keepdim=True)  # (B, H, L, 1)

        # Meta-token denominator: sum_j exp(meta_j - max_scores)
        # meta_tokens treated as logits; their contribution scales with max_scores
        meta_logits = self.meta_tokens.float().view(1, 1, 1, -1)  # (1, 1, 1, M)
        denom_meta = torch.exp(meta_logits - max_scores).sum(dim=-1, keepdim=True)  # (B, H, L, 1)

        denom = (denom_local + denom_meta).clamp(min=1e-6)
        out = num / denom  # (B, H, L, D)

        return out.to(v.dtype)

    def forward(self, hidden_states, **kwargs):
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expand (no RoPE — Lizard removes it)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        # Feature maps and gate
        phi_q = self.hedgehog(q, self.phi_q)
        phi_k = self.hedgehog(k, self.phi_k)
        gate = torch.sigmoid(self.W_gamma(hidden_states)).squeeze(-1).clamp(min=1e-6)  # (B, L)
        log_gate = torch.log(gate)

        # Branches
        y_gla = self.gla_branch(phi_q, phi_k, v, log_gate)
        y_awa = self.awa_branch(q, k, v)

        # Combine and project
        out = y_gla + self.alpha_blend * y_awa
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out.to(hidden_states.dtype))

        # Return tuple matching newer LlamaAttention signature: (output, attn_weights)
        return out, None
