"""
Lizard's hardware-aware GLA reparameterization (arXiv:2507.09025, Section 4).

The paper's trick, restated:

  The standard *parallel* form of gated linear attention is
      Y = ( ( (phi(Q) ⊙ C) (phi(K) / C)^T ) ⊙ M ) V
  where C is the row-cumulative product of gates (c_t = prod_{j<=t} gamma_j) and
  M is the causal mask. C underflows in bf16 (a long product of values < 1),
  which forces an fp32 fallback that is 2-3x slower and cannot use Tensor Cores.

  Lizard exploits that the Hedgehog feature map is *exponential*,
      phi(x) = [exp(xW) ⊕ exp(-xW)],
  so the cumulative gate can be absorbed into the exponent in log space:
      Q~ = [exp(QW + logC) ⊕ exp(-QW + logC)]
      K~ = [exp(KW - logC) ⊕ exp(-KW - logC)]
      Y  = ( (Q~ K~^T) ⊙ M ) V
  This is a plain GEMM (maps to mma.sync Tensor Cores), runs in bf16 with no
  fallback, and needs no custom kernel.

This file implements that reparameterization and verifies it equals the naive
gated recurrence. Shapes:
    x_q, x_k : (B, H, L, d)   raw per-head inputs to the feature map
    W        : (d, F)         hedgehog projection (phi maps d -> 2F)
    v        : (B, H, L, V)
    gamma    : (B, H, L)      scalar gate in (0,1)
"""

import torch


def _hedgehog(xW):
    """phi via an already-projected xW: [exp(xW) ⊕ exp(-xW)] -> (..., 2F).

    (The paper writes softmax-normalized variants elsewhere; the Section-4
    derivation uses the raw exp form, which is what makes the log-space absorb
    exact. We match Section 4.)
    """
    return torch.cat([torch.exp(xW), torch.exp(-xW)], dim=-1)


def gla_naive_hedgehog(x_q, x_k, v, gamma, W):
    """Ground-truth gated linear attention with the exp hedgehog feature map.

    Sequential recurrence on the *feature-mapped* q, k:
        phi_q = [exp(QW) ⊕ exp(-QW)],  phi_k likewise
        S_t = gamma_t S_{t-1} + phi_k(t)^T v_t
        y_t = phi_q(t) S_t
    """
    B, H, L, d = x_q.shape
    Vd = v.shape[-1]
    phi_q = _hedgehog(x_q @ W)               # (B,H,L,2F)
    phi_k = _hedgehog(x_k @ W)
    F2 = phi_q.shape[-1]
    S = torch.zeros(B, H, F2, Vd, dtype=v.dtype, device=v.device)
    out = torch.empty(B, H, L, Vd, dtype=v.dtype, device=v.device)
    for t in range(L):
        g_t = gamma[:, :, t].view(B, H, 1, 1)
        kt = phi_k[:, :, t].unsqueeze(-1)    # (B,H,2F,1)
        vt = v[:, :, t].unsqueeze(-2)        # (B,H,1,V)
        S = g_t * S + kt @ vt
        out[:, :, t] = (phi_q[:, :, t].unsqueeze(-2) @ S).squeeze(-2)
    return out


def gla_lizard_reparam(x_q, x_k, v, gamma, W):
    """Lizard's log-space GEMM form. No scan, no custom kernel.

    logC_t = cumsum_t log(gamma)   (row-cumulative log gate)
    Q~ = [exp(QW + logC) ⊕ exp(-QW + logC)]
    K~ = [exp(KW - logC) ⊕ exp(-KW - logC)]
    Y  = ( tril(Q~ K~^T) ) V
    """
    B, H, L, d = x_q.shape
    QW = x_q @ W                                          # (B,H,L,F)
    KW = x_k @ W
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1)  # (B,H,L)
    logC = logC.unsqueeze(-1)                             # (B,H,L,1)

    Q_tilde = torch.cat([torch.exp(QW + logC), torch.exp(-QW + logC)], dim=-1)
    K_tilde = torch.cat([torch.exp(KW - logC), torch.exp(-KW - logC)], dim=-1)

    scores = Q_tilde @ K_tilde.transpose(-2, -1)          # (B,H,L,L)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=v.device))
    scores = scores.masked_fill(~causal, 0.0)
    return scores @ v                                     # (B,H,L,V)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, d, Fdim, V = 2, 3, 128, 8, 6, 10
    x_q = torch.randn(B, H, L, d, dtype=torch.float64)
    x_k = torch.randn(B, H, L, d, dtype=torch.float64)
    v = torch.randn(B, H, L, V, dtype=torch.float64)
    W = torch.randn(d, Fdim, dtype=torch.float64) * 0.5
    gamma = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float64)) * 0.1 + 0.9

    ref = gla_naive_hedgehog(x_q, x_k, v, gamma, W)
    out = gla_lizard_reparam(x_q, x_k, v, gamma, W)
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"Lizard reparam vs naive recurrence:  rel_err = {rel:.3e}  "
          f"{'OK' if rel < 1e-9 else 'FAIL'}")

    # Show the numerical-stability point: in float32 with a longer sequence and
    # smaller gates, the *raw* cumulative product C underflows, but logC does not.
    Lbig = 512
    gamma_small = torch.full((1, 1, Lbig), 0.95, dtype=torch.float32)
    C_raw = torch.cumprod(gamma_small, dim=-1)            # the unstable quantity
    logC_stable = torch.cumsum(torch.log(gamma_small), dim=-1)
    print(f"\nAt L={Lbig}, gamma=0.95, float32:")
    print(f"  min raw C        = {C_raw.min().item():.3e}   "
          f"(underflows toward 0; reciprocal 1/C explodes)")
    print(f"  min logC         = {logC_stable.min().item():.3e}   "
          f"(well-behaved; stays in fp range)")
