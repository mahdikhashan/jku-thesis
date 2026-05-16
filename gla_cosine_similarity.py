"""
Numerical comparison of your FLA-based gla_branch against a slow PyTorch reference.

The reference implements the paper's formula exactly:
  y_i = phi(q_i) · sum_t [prod_l>t gate_l] phi(k_t) v_t^T
        / phi(q_i) · sum_t [prod_l>t gate_l] phi(k_t)

If your gla_branch produces near-identical output to the reference (cosine ~0.999),
GLA is implemented correctly and the cosine-similarity issue is elsewhere.
If they diverge, GLA is the bug.
"""

import torch
import torch.nn.functional as F
from types import SimpleNamespace

from lizard_attention import LizardAttention


# ---- Test configuration ----
# Use realistic shapes that match your actual model:
#   B=1, H=32 (Llama-3.2-1B num_heads), L=128, K=256 (2*FEATURE_DIM), D=64 (head_dim)
# Start with a smaller smoke test, then scale up.
TEST_CONFIGS = [
    # (B, H, L, K, D, name)
    (1, 4, 32, 8, 4, "tiny — smoke test"),
    (1, 32, 128, 256, 64, "realistic — matches your model"),
]


# ---- Reference implementation (paper's formula, slow but correct) ----
def reference_gla(phi_q, phi_k, v, log_gate):
    """Normalized gated linear attention, straight from paper Section 3.1.
    
    phi_q, phi_k: (B, H, L, K)
    v:            (B, H, L, D)
    log_gate:     (B, L)  -- shared across heads (Lizard's scalar gate)
    Returns:      (B, H, L, D)
    """
    B, H, L, K = phi_q.shape
    
    # Compute in fp32 for stability
    phi_q_f = phi_q.float()
    phi_k_f = phi_k.float()
    v_f = v.float()
    
    cum = log_gate.cumsum(dim=-1)                      # (B, L)
    diff = cum.unsqueeze(-1) - cum.unsqueeze(-2)       # (B, L, L)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=phi_q.device))
    diff = diff.masked_fill(~causal, float('-inf'))
    G = torch.exp(diff)                                # (B, L, L), bounded in (0, 1]
    
    scores = torch.matmul(phi_q_f, phi_k_f.transpose(-2, -1))  # (B, H, L, L)
    scores = scores * G.unsqueeze(1)                          # broadcast over heads
    
    num = torch.matmul(scores, v_f)                            # (B, H, L, D)
    denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (B, H, L, 1)
    
    return (num / denom).to(v.dtype)


# ---- Helper: build a minimal LizardAttention without needing a real Llama config ----
def make_lizard_attention(hidden_size, num_heads):
    """Construct a LizardAttention with minimal config for branch testing."""
    config = SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,  # no GQA for the standalone test
    )
    attn = LizardAttention(config, layer_idx=0).cuda().to(torch.bfloat16)
    return attn


# ---- Run tests ----
def run_one(B, H, L, K, D, name):
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  B={B}, H={H}, L={L}, K={K}, D={D}")
    print(f"{'='*70}")
    
    torch.manual_seed(0)
    
    # Build inputs. phi_q/phi_k are post-Hedgehog (non-negative).
    # Add a small floor so we never have exact zeros.
    phi_q = (torch.rand(B, H, L, K, device='cuda') + 0.1).to(torch.bfloat16)
    phi_k = (torch.rand(B, H, L, K, device='cuda') + 0.1).to(torch.bfloat16)
    v = torch.randn(B, H, L, D, device='cuda', dtype=torch.bfloat16)
    
    # Two gate regimes worth testing
    test_gates = {
        "gate=1 (no decay)":        torch.zeros(B, L, dtype=torch.float32, device='cuda'),
        "gate=0.5 (moderate decay)": torch.full((B, L), -0.693, dtype=torch.float32, device='cuda'),
    }
    
    # Build a Lizard attention module sized to match this test.
    # The gla_branch only uses the projections that operate post-Hedgehog,
    # so the hidden_size/num_heads here mostly affect which constants are read.
    hidden_size = H * D
    attn = make_lizard_attention(hidden_size, H)
    attn.eval()
    
    for gate_name, log_gate in test_gates.items():
        with torch.no_grad():
            # Reference
            out_ref = reference_gla(phi_q, phi_k, v, log_gate)
            
            # Your impl. gla_branch signature: (phi_q, phi_k, v, log_gate)
            out_fla = attn.gla_branch(phi_q, phi_k, v, log_gate)
        
        # Both to fp32 for fair comparison
        a = out_fla.float().flatten()
        b = out_ref.float().flatten()
        
        max_abs = (a - b).abs().max().item()
        cos = F.cosine_similarity(a, b, dim=0).item()
        norm_ratio = a.norm().item() / b.norm().clamp(min=1e-8).item()
        rel_err = (a - b).norm().item() / b.norm().clamp(min=1e-8).item()
        
        verdict = "MATCH" if cos > 0.99 and abs(norm_ratio - 1.0) < 0.05 else "MISMATCH"
        
        print(f"\n  {gate_name}:")
        print(f"    max abs diff: {max_abs:.4e}")
        print(f"    cosine sim:   {cos:.6f}")
        print(f"    norm ratio:   {norm_ratio:.6f}  (your impl / reference)")
        print(f"    relative err: {rel_err:.4f}")
        print(f"    verdict:      {verdict}")


def main():
    for cfg in TEST_CONFIGS:
        run_one(*cfg)
    
    print(f"\n{'='*70}")
    print("Interpretation:")
    print("  cosine > 0.99 AND norm_ratio in [0.95, 1.05]: GLA is correct")
    print("  cosine 0.7-0.99 OR norm_ratio off:            partial match, FLA has subtle differences")
    print("  cosine < 0.7:                                  GLA is fundamentally miscomputing")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
