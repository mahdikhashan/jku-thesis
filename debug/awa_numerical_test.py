"""
Numerical comparison of your FA2-based awa_branch against a slow PyTorch reference.

Tests two things:
1. AWA without meta tokens (just sliding window softmax) — confirms the FA2 + window invocation is correct
2. AWA with meta tokens (full Lizard formulation) — confirms the LSE-based denominator rescale is correct

If cosine ~0.999 for both, AWA is implemented correctly and the eval gap is elsewhere.
If either fails, we've found the bug.
"""

import math
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from lizard_attention import LizardAttention, WINDOW_SIZE, NUM_META_TOKENS


TEST_CONFIGS = [
    # (B, H, L, D, name)
    (1,  4,  64,  16, "tiny — smoke test"),
    (1, 32, 256,  64, "realistic — matches your model"),
    (1, 32, 512,  64, "long — sequence > window size"),  # tests window correctness
]


# ---- Reference: sliding-window causal softmax attention (no meta tokens) ----
def reference_awa_no_meta(q, k, v, window_size):
    """Paper's AWA without meta-token denominator. Plain sliding-window softmax.
    
    Args:
        q, k, v: (B, H, L, D)
        window_size: int, size of causal window
    Returns:
        (B, H, L, D)
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    # Sliding causal window mask: position i attends to [max(0, i-W+1), i]
    idx = torch.arange(L, device=q.device)
    valid = (idx.unsqueeze(0) <= idx.unsqueeze(1)) & \
            ((idx.unsqueeze(1) - idx.unsqueeze(0)) < window_size)
    # valid[i, j] = True iff j <= i and (i - j) < W
    
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B, H, L, L)
    scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Standard softmax over the unmasked region per query
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())  # (B, H, L, D)
    return out.to(v.dtype)


# ---- Reference: sliding-window with meta-token denominator (paper formula) ----
def reference_awa_with_meta(q, k, v, window_size, meta_tokens):
    """Full Lizard AWA per paper Section 3.1:
    
      y_i = sum_{t in window} exp(q_i · k_t / sqrt(d)) v_t
            / [sum_j exp(t_j) + sum_{t in window} exp(q_i · k_t / sqrt(d))]
    
    where t_j are the meta-token logits.
    
    Args:
        q, k, v: (B, H, L, D)
        window_size: int
        meta_tokens: (M,) logits for M meta tokens
    Returns:
        (B, H, L, D)
    """
    B, H, L, D = q.shape
    scale = 1.0 / math.sqrt(D)
    
    idx = torch.arange(L, device=q.device)
    valid = (idx.unsqueeze(0) <= idx.unsqueeze(1)) & \
            ((idx.unsqueeze(1) - idx.unsqueeze(0)) < window_size)
    
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B, H, L, L)
    scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Numerator: sum of exp(score) * v over window
    # Denominator: sum of exp(score) + sum of exp(meta_tokens)
    # Compute stably via max-subtraction
    max_scores = scores.max(dim=-1, keepdim=True).values  # (B, H, L, 1)
    # Replace -inf max with 0 (happens when window is empty, though shouldn't here)
    max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)
    
    exp_scores = torch.exp(scores - max_scores)  # (B, H, L, L), zero where masked
    exp_scores = exp_scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), 0.0)
    
    num = torch.matmul(exp_scores, v.float())                # (B, H, L, D)
    denom_local = exp_scores.sum(dim=-1, keepdim=True)       # (B, H, L, 1)
    
    # Meta-token contribution: sum_j exp(meta_j - max_scores)
    # Broadcasting: meta_tokens (M,) -> (1, 1, 1, M); max_scores (B, H, L, 1)
    meta_logits = meta_tokens.float().view(1, 1, 1, -1)      # (1, 1, 1, M)
    denom_meta = torch.exp(meta_logits - max_scores).sum(dim=-1, keepdim=True)  # (B, H, L, 1)
    
    denom = (denom_local + denom_meta).clamp(min=1e-6)
    return (num / denom).to(v.dtype)


# ---- Helper: build a minimal LizardAttention ----
def make_lizard_attention(hidden_size, num_heads, meta_init_value=0.0):
    """Construct a LizardAttention with explicit meta_token init."""
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,  # no GQA for the test
    )
    attn = LizardAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
    # Set meta tokens to a known value
    with torch.no_grad():
        attn.meta_tokens.data.fill_(meta_init_value)
    return attn


# ---- Run tests ----
def run_one(B, H, L, D, name):
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  B={B}, H={H}, L={L}, D={D}, W={WINDOW_SIZE}, M={NUM_META_TOKENS}")
    print(f"{'='*70}")
    
    torch.manual_seed(0)
    
    # Realistic-ish inputs (post-projection q/k/v have std ~1)
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    
    hidden_size = H * D
    
    # ---- Subtest 1: meta tokens = -inf (effectively disabled) ----
    # When meta_tokens are very negative, exp(meta) ~ 0, denominator reduces to just local
    # So this should match the "no meta" reference exactly
    print("\n  Subtest 1: meta_tokens disabled (set to -1e9)")
    attn = make_lizard_attention(hidden_size, H, meta_init_value=-1e9)
    attn.eval()
    
    with torch.no_grad():
        out_yours = attn.awa_branch(q, k, v)
        out_ref = reference_awa_no_meta(q, k, v, WINDOW_SIZE)
    
    a = out_yours.float().flatten()
    b = out_ref.float().flatten()
    
    max_abs = (a - b).abs().max().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    norm_ratio = a.norm().item() / b.norm().clamp(min=1e-8).item()
    rel_err = (a - b).norm().item() / b.norm().clamp(min=1e-8).item()
    verdict = "MATCH" if cos > 0.99 and abs(norm_ratio - 1.0) < 0.05 else "MISMATCH"
    
    print(f"    max abs diff: {max_abs:.4e}")
    print(f"    cosine sim:   {cos:.6f}")
    print(f"    norm ratio:   {norm_ratio:.6f}")
    print(f"    relative err: {rel_err:.4f}")
    print(f"    verdict:      {verdict}")
    
    # ---- Subtest 2: meta tokens at default zero init ----
    print("\n  Subtest 2: meta_tokens = 0 (default init, contributes M to denominator)")
    attn = make_lizard_attention(hidden_size, H, meta_init_value=0.0)
    attn.eval()
    
    with torch.no_grad():
        out_yours = attn.awa_branch(q, k, v)
        out_ref = reference_awa_with_meta(q, k, v, WINDOW_SIZE, attn.meta_tokens.data)
    
    a = out_yours.float().flatten()
    b = out_ref.float().flatten()
    
    max_abs = (a - b).abs().max().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    norm_ratio = a.norm().item() / b.norm().clamp(min=1e-8).item()
    rel_err = (a - b).norm().item() / b.norm().clamp(min=1e-8).item()
    verdict = "MATCH" if cos > 0.99 and abs(norm_ratio - 1.0) < 0.05 else "MISMATCH"
    
    print(f"    max abs diff: {max_abs:.4e}")
    print(f"    cosine sim:   {cos:.6f}")
    print(f"    norm ratio:   {norm_ratio:.6f}")
    print(f"    relative err: {rel_err:.4f}")
    print(f"    verdict:      {verdict}")
    
    # ---- Subtest 3: meta tokens at trained value (~0.25) ----
    print("\n  Subtest 3: meta_tokens = 0.25 (matches trained values)")
    attn = make_lizard_attention(hidden_size, H, meta_init_value=0.25)
    attn.eval()
    
    with torch.no_grad():
        out_yours = attn.awa_branch(q, k, v)
        out_ref = reference_awa_with_meta(q, k, v, WINDOW_SIZE, attn.meta_tokens.data)
    
    a = out_yours.float().flatten()
    b = out_ref.float().flatten()
    
    max_abs = (a - b).abs().max().item()
    cos = F.cosine_similarity(a, b, dim=0).item()
    norm_ratio = a.norm().item() / b.norm().clamp(min=1e-8).item()
    rel_err = (a - b).norm().item() / b.norm().clamp(min=1e-8).item()
    verdict = "MATCH" if cos > 0.99 and abs(norm_ratio - 1.0) < 0.05 else "MISMATCH"
    
    print(f"    max abs diff: {max_abs:.4e}")
    print(f"    cosine sim:   {cos:.6f}")
    print(f"    norm ratio:   {norm_ratio:.6f}")
    print(f"    relative err: {rel_err:.4f}")
    print(f"    verdict:      {verdict}")


def main():
    for cfg in TEST_CONFIGS:
        run_one(*cfg)
    
    print(f"\n{'='*70}")
    print("Interpretation:")
    print("  All MATCH → AWA branch is correctly implemented; bug is elsewhere")
    print("  Subtest 1 MISMATCH → FA2 windowed invocation is wrong")
    print("    (window size, causal flag, or transpose between (B,H,L,D)/(B,L,H,D))")
    print("  Subtest 2 or 3 MISMATCH but Subtest 1 OK → meta-token rescale is wrong")
    print("    (LSE base, log_total computation, or rescale broadcasting)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
