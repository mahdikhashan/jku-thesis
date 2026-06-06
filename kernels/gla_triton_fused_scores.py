"""
Option 1 — Fused reparameterization + score GEMM (Triton).

The paper reduces gated linear attention to a GEMM:
    Q~ = [exp(QW + logC) ⊕ exp(-QW + logC)]
    K~ = [exp(KW - logC) ⊕ exp(-KW - logC)]
    Y  = ( tril(Q~ K~^T) ) V

The straightforward PyTorch version (gla_lizard_reparam.py) materializes the
2F-wide tensors Q~ and K~ in HBM via exp/cat, then calls matmul. This kernel
fuses the reparameterization into the prologue of the score matmul: it loads
QW, KW, and logC tiles, forms exp(±· + logC) on-chip, and accumulates the
score block directly — Q~ and K~ are never written to global memory.

What is and isn't fused here:
  - FUSED:   the exp(±QW + logC) / exp(±KW - logC) reparam, and the Q~ K~^T
             score GEMM with the causal mask.
  - NOT:     the input projections QW = x_q @ W and KW = x_k @ W (done in torch
             beforehand — they are ordinary GEMMs cuBLAS handles well), and the
             final scores @ V (a second GEMM, left to torch here; Option 2 fuses
             the whole pipeline including this).

This is the honest, modest Triton contribution: it removes the HBM round-trip
for the doubled-width feature tensors. Whether it beats the unfused cuBLAS path
depends on your GPU and shapes — measure with bench (compare to
gla_lizard_reparam).

NOT RUN ON GPU IN AUTHORING. Validate with `python gla_triton_fused_scores.py
--check` on your CUDA GPU; it compares against the CPU-verified reference.

Assumes F (feature_dim) <= 128 so a QW/KW row loads whole. head_dim V tiled by
BLOCK_V if needed; here loaded whole for V <= 128 (Llama-3.2-1B: V=64).
"""

import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_score_av_kernel(
    QW, KW, LOGC, V, OUT,
    stride_qb, stride_qt, stride_qf,      # QW/KW: (BH, L, F)
    stride_cb, stride_ct,                 # LOGC:  (BH, L)
    stride_vb, stride_vt, stride_vv,      # V:     (BH, L, Vd)
    stride_ob, stride_ot, stride_ov,      # OUT:   (BH, L, Vd)
    L, F: tl.constexpr, Vd: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per (bh, query-block). Streams key-blocks, forming Q~,K~
    on-chip and accumulating O = (tril(Q~K~^T)) V for this query block.

    Q~ row for query i: [exp(qw_i + logc_i), exp(-qw_i + logc_i)]   (width 2F)
    K~ row for key  j: [exp(kw_j - logc_j), exp(-kw_j - logc_j)]
    score_ij = Q~_i . K~_j
             = exp(logc_i + logc_j) * [ exp(qw_i)·exp(kw_j) + exp(-qw_i)·exp(-kw_j) ]
    We compute it via the explicit 2F dot to stay close to the paper's form.
    """
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # query rows
    offs_f = tl.arange(0, F)
    offs_v = tl.arange(0, Vd)

    qbase = QW + pid_bh * stride_qb
    kbase = KW + pid_bh * stride_qb
    cbase = LOGC + pid_bh * stride_cb
    vbase = V + pid_bh * stride_vb

    # Load this query block's QW (BLOCK_M, F) and logC (BLOCK_M,)
    m_mask = offs_m < L
    qw = tl.load(qbase + offs_m[:, None] * stride_qt + offs_f[None, :] * stride_qf,
                 mask=m_mask[:, None], other=0.0)
    logc_m = tl.load(cbase + offs_m * stride_ct, mask=m_mask, other=0.0)

    # On-chip Q~ halves: (BLOCK_M, F) each
    qpos = tl.exp(qw + logc_m[:, None])      # exp(qw + logc)
    qneg = tl.exp(-qw + logc_m[:, None])     # exp(-qw + logc)

    acc = tl.zeros((BLOCK_M, Vd), dtype=tl.float32)

    # Stream key blocks 0..pid_m (causal: keys beyond query block are masked out)
    for start_n in range(0, (pid_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L
        kw = tl.load(kbase + offs_n[:, None] * stride_qt + offs_f[None, :] * stride_qf,
                     mask=n_mask[:, None], other=0.0)
        logc_n = tl.load(cbase + offs_n * stride_ct, mask=n_mask, other=0.0)

        kpos = tl.exp(kw - logc_n[:, None])      # (BLOCK_N, F)
        kneg = tl.exp(-kw - logc_n[:, None])

        # score = qpos @ kpos^T + qneg @ kneg^T   -> (BLOCK_M, BLOCK_N)
        score = tl.dot(qpos, tl.trans(kpos)) + tl.dot(qneg, tl.trans(kneg))

        # causal mask: keep key j <= query i
        causal = offs_m[:, None] >= offs_n[None, :]
        score = tl.where(causal & n_mask[None, :], score, 0.0)

        # accumulate score @ V_block
        vblk = tl.load(vbase + offs_n[:, None] * stride_vt + offs_v[None, :] * stride_vv,
                       mask=n_mask[:, None], other=0.0)
        acc += tl.dot(score.to(vblk.dtype), vblk)

    obase = OUT + pid_bh * stride_ob
    tl.store(obase + offs_m[:, None] * stride_ot + offs_v[None, :] * stride_ov,
             acc, mask=m_mask[:, None])


def gla_fused_scores(x_q, x_k, v, gamma, W, block_m=64, block_n=64):
    """Option-1 path: torch projections + fused (reparam, scores, scores@V) kernel.

    x_q, x_k : (B,H,L,d)   v : (B,H,L,Vd)   gamma : (B,H,L)   W : (d,F)
    """
    B, H, L, d = x_q.shape
    Vd = v.shape[-1]
    F = W.shape[-1]

    QW = (x_q @ W).reshape(B * H, L, F).contiguous()
    KW = (x_k @ W).reshape(B * H, L, F).contiguous()
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1).reshape(B * H, L).contiguous()
    vv = v.reshape(B * H, L, Vd).contiguous()
    out = torch.empty(B * H, L, Vd, device=x_q.device, dtype=torch.float32)

    grid = (B * H, triton.cdiv(L, block_m))
    _fused_score_av_kernel[grid](
        QW, KW, logC, vv, out,
        QW.stride(0), QW.stride(1), QW.stride(2),
        logC.stride(0), logC.stride(1),
        vv.stride(0), vv.stride(1), vv.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L, F=F, Vd=Vd, BLOCK_M=block_m, BLOCK_N=block_n,
    )
    return out.view(B, H, L, Vd).to(v.dtype)


def _check():
    from gla_lizard_reparam import gla_naive_hedgehog
    torch.manual_seed(0)
    B, H, L, d, F, Vd = 2, 3, 192, 8, 16, 24
    dev = "cuda"
    x_q = torch.randn(B, H, L, d, device=dev)
    x_k = torch.randn(B, H, L, d, device=dev)
    v = torch.randn(B, H, L, Vd, device=dev)
    W = torch.randn(d, F, device=dev) * 0.3
    gamma = torch.sigmoid(torch.randn(B, H, L, device=dev)) * 0.1 + 0.9

    ref = gla_naive_hedgehog(x_q.double(), x_k.double(), v.double(),
                             gamma.double(), W.double()).float()
    out = gla_fused_scores(x_q, x_k, v, gamma, W)
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"[option1] fused scores vs naive ref:  rel_err={rel:.3e}  "
          f"{'OK' if rel < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        print("Run with --check on a CUDA GPU, or import gla_fused_scores.")
