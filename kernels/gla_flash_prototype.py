"""
CPU prototype of the Option-2 flash-style tiled computation, to verify the
block decomposition BEFORE writing it in Triton.

Option 2 computes Y = (tril(Q~ K~^T)) V without ever forming the full L x L
score matrix, FlashAttention-style:

  For each query block i, stream key/value blocks j <= i:
    - off-diagonal (j < i): the whole block is causally valid -> dense
        O_i += (Q~_i K~_j^T) V_j
    - diagonal   (j == i): apply the within-block causal mask
        O_i += (tril(Q~_i K~_i^T)) V_i

Because linear attention has no softmax, there is no running max / running
denominator to track — accumulation is a plain sum over key blocks. (The
normalization Lizard uses is a separate division handled by running the same
pipeline with V = ones; omitted here, as in the other reference files, since
the numerator is the expensive part.)

This file verifies the tiled sum equals the naive recurrence, so the Triton
kernel in gla_triton_flash.py can transcribe a known-correct decomposition.
"""

import torch
from gla_lizard_reparam import gla_naive_hedgehog, _hedgehog


def gla_flash_tiled_cpu(x_q, x_k, v, gamma, W, block=32):
    """Tiled linear attention matching the Option-2 Triton kernel's structure."""
    B, H, L, d = x_q.shape
    Vd = v.shape[-1]

    QW = x_q @ W
    KW = x_k @ W
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1).unsqueeze(-1)
    Q = torch.cat([torch.exp(QW + logC), torch.exp(-QW + logC)], dim=-1)   # (B,H,L,2F)
    K = torch.cat([torch.exp(KW - logC), torch.exp(-KW - logC)], dim=-1)

    out = torch.zeros(B, H, L, Vd, dtype=v.dtype, device=v.device)
    n_blk = (L + block - 1) // block

    for bi in range(n_blk):
        qs, qe = bi * block, min((bi + 1) * block, L)
        Qi = Q[:, :, qs:qe]                                  # (B,H,bm,2F)
        acc = torch.zeros(B, H, qe - qs, Vd, dtype=v.dtype, device=v.device)
        for bj in range(bi + 1):
            ks, ke = bj * block, min((bj + 1) * block, L)
            Kj = K[:, :, ks:ke]
            Vj = v[:, :, ks:ke]
            s = Qi @ Kj.transpose(-2, -1)                    # (B,H,bm,bn)
            if bj == bi:
                # diagonal block: causal mask within block
                qidx = torch.arange(qs, qe, device=v.device)[:, None]
                kidx = torch.arange(ks, ke, device=v.device)[None, :]
                s = s.masked_fill(kidx > qidx, 0.0)
            acc = acc + s @ Vj
        out[:, :, qs:qe] = acc
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, L, d, F, Vd = 2, 3, 200, 8, 6, 10
    x_q = torch.randn(B, H, L, d, dtype=torch.float64)
    x_k = torch.randn(B, H, L, d, dtype=torch.float64)
    v = torch.randn(B, H, L, Vd, dtype=torch.float64)
    W = torch.randn(d, F, dtype=torch.float64) * 0.5
    gamma = torch.sigmoid(torch.randn(B, H, L, dtype=torch.float64)) * 0.1 + 0.9

    ref = gla_naive_hedgehog(x_q, x_k, v, gamma, W)
    for blk in (8, 32, 64, 128):
        out = gla_flash_tiled_cpu(x_q, x_k, v, gamma, W, block=blk)
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        print(f"[option2 cpu] block={blk:4d}  rel_err={rel:.3e}  "
              f"{'OK' if rel < 1e-9 else 'FAIL'}")
