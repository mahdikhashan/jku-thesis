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
        """Sliding-window softmax attention via FA2, with meta-token denominator.

        q, k, v: (B, H, L, head_dim)
        Returns: (B, H, L, head_dim)

        Math:
            FA2 returns      out = numerator / denom_local
            We want    out_corrected = numerator / (denom_local + denom_meta)
                                     = out * denom_local / (denom_local + denom_meta)
            denom_local = exp(lse)                  (FA2 returns lse directly)
            denom_meta  = sum_j exp(meta_token_j)
            rescale     = exp(lse - logaddexp(lse, log_sum_meta))  ∈ (0, 1]
        """
        B, H, L, D = q.shape

        # FA2 expects (B, L, H, D)
        q_ = q.transpose(1, 2).contiguous()
        k_ = k.transpose(1, 2).contiguous()
        v_ = v.transpose(1, 2).contiguous()

        # Causal sliding window of size W: position i attends to [i-W+1, i]
        out, lse, _ = flash_attn_func(
            q_, k_, v_,
            causal=True,
            window_size=(WINDOW_SIZE - 1, 0),
            return_attn_probs=True,
        )
        # out: (B, L, H, D)
        # lse: (B, H, L) — log of softmax denominator over the window (fp32)

        # Meta-token denominator correction
        log_sum_meta = torch.logsumexp(self.meta_tokens.float(), dim=0)  # scalar
        log_total = torch.logaddexp(lse.float(), log_sum_meta)            # (B, H, L)
        rescale = torch.exp(lse.float() - log_total)                      # (B, H, L), in (0, 1]

        # Broadcast rescale (B, H, L) -> (B, L, H, 1) to match out (B, L, H, D)
        out = out * rescale.transpose(1, 2).unsqueeze(-1).to(out.dtype)
        return out.transpose(1, 2).contiguous()

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
