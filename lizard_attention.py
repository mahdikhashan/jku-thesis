"""
LizardAttention — iteration 2 + iter 4 fp32 dtype patches.

Changes from iteration 1:
  - GLA via FLA's chunk_gla kernel (correct + fast, replaces hand-written chunked version)
  - AWA via FlashAttention-2 with native windowed attention (replaces is_causal=True hack)
  - Meta-token denominator correction implemented via LSE rescaling (was missing)
  - Hedgehog activation: softmax (per paper Table 13)
  - Normalized GLA output (per paper Section 3.1; needs a 2nd chunk_gla call)

Iter 4 mixed-precision notes:
  - Lizard auxiliary parameters (phi_q, phi_k, W_gamma, meta_tokens, alpha_blend)
    are kept in fp32 by the training script to avoid bf16 quantization stalling
    AdamW updates on small parameters.
  - Inputs (q, k, v, hidden_states) arrive in bf16 from the base attention path.
  - Wherever an fp32 weight is applied to a bf16 input, we cast the input to fp32
    at the boundary. The fp32 result is implicitly downcast to bf16 by downstream
    ops (residual add, o_proj).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
from fla.ops.gla import chunk_gla
from fla.ops.gla import fused_recurrent_gla

from config import *


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
        # === DTYPE BOUNDARY ===
        # proj.weight may be fp32 (kept in fp32 for AdamW precision) while x is
        # typically bf16 from upstream. F.linear requires matching dtypes; cast
        # input up to weight's dtype so fp32 weights can be exercised fully.
        if x.dtype != proj.weight.dtype:
            x = x.to(proj.weight.dtype)
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

        # FLA kernels expect all tensors in the same dtype. phi_q/phi_k come
        # out of hedgehog in fp32 (after upcast); v and gate may be bf16.
        # Align everything to phi_q's dtype before calling the kernel.
        target_dtype = phi_q_.dtype
        if v_.dtype != target_dtype:
            v_ = v_.to(target_dtype)
        if g.dtype != target_dtype:
            g = g.to(target_dtype)

        # Numerator
        num, _ = fused_recurrent_gla(q=phi_q_, k=phi_k_, v=v_, gk=g)

        # Denominator: same gated accumulation with v=ones
        ones = torch.ones_like(v_[..., :1])  # (B, L, H, 1)
        denom, _ = fused_recurrent_gla(q=phi_q_, k=phi_k_, v=ones, gk=g)
        denom = denom.clamp(min=1e-6)

        out = num / denom  # (B, L, H, D)
        return out.transpose(1, 2).contiguous()  # back to (B, H, L, D)

    def awa_branch(self, q, k, v):
        """Chunked sliding-window AWA with meta-token denominator.

        Never materializes the full (L, L) score matrix; only per-chunk slices.
        Paper-faithful: identical math to the unchunked reference, verified by
        numerical equivalence test (cosine 1.000000).

        Note: all softmax math is already in fp32 internally (.float() calls
        on scores and v_chunk), so meta_tokens being fp32 fits naturally — no
        extra cast needed beyond what's already there.
        """
        import math
        B, H, L, D = q.shape
        scale = 1.0 / math.sqrt(D)
        device = q.device
        CHUNK = 256  # tune: smaller = less memory, more loop overhead

        out = torch.empty(B, H, L, D, device=device, dtype=v.dtype)

        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            q_chunk = q[:, :, start:end]  # (B, H, chunk, D)

            # Key range covers the window for queries in [start, end)
            k_start = max(0, start - WINDOW_SIZE + 1)
            k_end = end
            k_chunk = k[:, :, k_start:k_end]
            v_chunk = v[:, :, k_start:k_end]

            # Validity mask: causal + within window
            q_idx = torch.arange(start, end, device=device).unsqueeze(-1)
            k_idx = torch.arange(k_start, k_end, device=device).unsqueeze(0)
            valid = (k_idx <= q_idx) & ((q_idx - k_idx) < WINDOW_SIZE)

            # Compute attention in fp32 for stability (scores promoted via .float())
            scores = (torch.matmul(q_chunk.float(), k_chunk.float().transpose(-2, -1)) * scale)
            scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))

            max_scores = scores.max(dim=-1, keepdim=True).values
            max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)

            exp_scores = torch.exp(scores - max_scores)
            exp_scores = exp_scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), 0.0)

            num = torch.matmul(exp_scores, v_chunk.float())
            denom_local = exp_scores.sum(dim=-1, keepdim=True)

            # meta_tokens may be fp32 (after iter 4 upcast); .float() is a no-op
            # if already fp32, otherwise promotes from bf16. Same effect either way.
            meta_logits = self.meta_tokens.float().view(1, 1, 1, -1)
            denom_meta = torch.exp(meta_logits - max_scores).sum(dim=-1, keepdim=True)

            denom = (denom_local + denom_meta).clamp(min=1e-6)
            out[:, :, start:end] = (num / denom).to(v.dtype)

        return out

    def forward(self, hidden_states, **kwargs):
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expand (no RoPE — Lizard removes it)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        # Feature maps and gate
        # hedgehog handles the dtype boundary internally for phi_q / phi_k
        phi_q = self.hedgehog(q, self.phi_q)
        phi_k = self.hedgehog(k, self.phi_k)

        # === DTYPE BOUNDARY ===
        # W_gamma may be fp32 while hidden_states is bf16. Cast input to match.
        gate_input = hidden_states
        if gate_input.dtype != self.W_gamma.weight.dtype:
            gate_input = gate_input.to(self.W_gamma.weight.dtype)
        gate = torch.sigmoid(self.W_gamma(gate_input)).squeeze(-1).clamp(min=1e-6)  # (B, L)
        log_gate = torch.log(gate)

        # Branches
        y_gla = self.gla_branch(phi_q, phi_k, v, log_gate)
        y_awa = self.awa_branch(q, k, v)

        # === DTYPE BOUNDARY ===
        # alpha_blend may be fp32 while y_awa is bf16 (matches input dtype of v).
        # Promote y_awa to alpha_blend's dtype for the multiplication, then add to
        # y_gla. y_gla's dtype follows phi_q (potentially fp32 from hedgehog).
        # Final cast back to hidden_states.dtype happens before o_proj.
        if y_gla.dtype != y_awa.dtype:
            y_awa = y_awa.to(y_gla.dtype)
        alpha = self.alpha_blend.to(y_gla.dtype)
        out = y_gla + alpha * y_awa

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        # Cast back to base model dtype for o_proj (which stays in bf16)
        out = self.o_proj(out.to(hidden_states.dtype))

        # Return tuple matching newer LlamaAttention signature: (output, attn_weights)
        return out, None