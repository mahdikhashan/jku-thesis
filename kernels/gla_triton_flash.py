"""
Option 2 — Full flash-style fused linear attention (Triton).

Computes Y = (tril(Q~ K~^T)) V in a single kernel without ever materializing
the L x L score matrix, transcribing the block decomposition verified in
gla_flash_prototype.py.

Per program (one query block per (bh, block)):
  - load QW block, logC block; form Q~ halves on-chip
  - stream key/value blocks j <= i:
        off-diagonal (j < i): dense    O += (Q~ K~_j^T) V_j
        diagonal     (j = i): masked   O += (tril(Q~ K~_i^T)) V_i
  - no running max / denominator: linear attention sums plainly over key blocks

This fuses the entire pipeline (reparam + scores + scores@V) and keeps the
score blocks in SRAM, so memory traffic is O(L) in activations rather than
O(L^2). This is the highest-performance form and the most likely to beat the
unfused cuBLAS path at long L — but also the one most sensitive to block-size
tuning and the hardest to get right.

NOT RUN ON GPU IN AUTHORING. Validate with `python gla_triton_flash.py --check`
on your CUDA GPU before trusting it; it compares against the CPU reference.
After --check passes, autotune BLOCK_M / BLOCK_N for your GPU.

Assumes F <= 128 and Vd <= 128 loaded whole (Llama-3.2-1B: F=128, Vd=64).
"""

import argparse
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["L", "F", "Vd"],
)
@triton.jit
def _flash_linear_kernel(
    QW, KW, LOGC, V, OUT,
    stride_qb, stride_qt, stride_qf,
    stride_cb, stride_ct,
    stride_vb, stride_vt, stride_vv,
    stride_ob, stride_ot, stride_ov,
    L, F: tl.constexpr, Vd: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_f = tl.arange(0, F)
    offs_v = tl.arange(0, Vd)

    qbase = QW + pid_bh * stride_qb
    kbase = KW + pid_bh * stride_qb
    cbase = LOGC + pid_bh * stride_cb
    vbase = V + pid_bh * stride_vb

    m_mask = offs_m < L
    qw = tl.load(qbase + offs_m[:, None] * stride_qt + offs_f[None, :] * stride_qf,
                 mask=m_mask[:, None], other=0.0)
    logc_m = tl.load(cbase + offs_m * stride_ct, mask=m_mask, other=0.0)
    qpos = tl.exp(qw + logc_m[:, None])
    qneg = tl.exp(-qw + logc_m[:, None])

    acc = tl.zeros((BLOCK_M, Vd), dtype=tl.float32)

    # last key index this query block can attend to
    hi = (pid_m + 1) * BLOCK_M
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < L
        kw = tl.load(kbase + offs_n[:, None] * stride_qt + offs_f[None, :] * stride_qf,
                     mask=n_mask[:, None], other=0.0)
        logc_n = tl.load(cbase + offs_n * stride_ct, mask=n_mask, other=0.0)
        kpos = tl.exp(kw - logc_n[:, None])
        kneg = tl.exp(-kw - logc_n[:, None])

        score = tl.dot(qpos, tl.trans(kpos)) + tl.dot(qneg, tl.trans(kneg))

        # causal mask only needed where blocks overlap the diagonal; applying
        # it unconditionally (key index <= query index) is correct for all
        # streamed blocks and cheap.
        causal = offs_m[:, None] >= offs_n[None, :]
        score = tl.where(causal & n_mask[None, :], score, 0.0)

        vblk = tl.load(vbase + offs_n[:, None] * stride_vt + offs_v[None, :] * stride_vv,
                       mask=n_mask[:, None], other=0.0)
        acc += tl.dot(score.to(vblk.dtype), vblk)

    obase = OUT + pid_bh * stride_ob
    tl.store(obase + offs_m[:, None] * stride_ot + offs_v[None, :] * stride_ov,
             acc, mask=m_mask[:, None])


def gla_flash_triton(x_q, x_k, v, gamma, W):
    """Option-2 path: torch projections + single fused flash-linear kernel."""
    B, H, L, d = x_q.shape
    Vd = v.shape[-1]
    F = W.shape[-1]

    QW = (x_q @ W).reshape(B * H, L, F).contiguous()
    KW = (x_k @ W).reshape(B * H, L, F).contiguous()
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1).reshape(B * H, L).contiguous()
    vv = v.reshape(B * H, L, Vd).contiguous()
    out = torch.empty(B * H, L, Vd, device=x_q.device, dtype=torch.float32)

    grid = lambda meta: (B * H, triton.cdiv(L, meta["BLOCK_M"]))
    _flash_linear_kernel[grid](
        QW, KW, logC, vv, out,
        QW.stride(0), QW.stride(1), QW.stride(2),
        logC.stride(0), logC.stride(1),
        vv.stride(0), vv.stride(1), vv.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        L, F=F, Vd=Vd,
    )
    return out.view(B, H, L, Vd).to(v.dtype)


def _check():
    from gla_lizard_reparam import gla_naive_hedgehog
    torch.manual_seed(0)
    B, H, L, d, F, Vd = 2, 3, 200, 8, 16, 24
    dev = "cuda"
    x_q = torch.randn(B, H, L, d, device=dev)
    x_k = torch.randn(B, H, L, d, device=dev)
    v = torch.randn(B, H, L, Vd, device=dev)
    W = torch.randn(d, F, device=dev) * 0.3
    gamma = torch.sigmoid(torch.randn(B, H, L, device=dev)) * 0.1 + 0.9

    ref = gla_naive_hedgehog(x_q.double(), x_k.double(), v.double(),
                             gamma.double(), W.double()).float()
    out = gla_flash_triton(x_q, x_k, v, gamma, W)
    rel = (out - ref).abs().max().item() / ref.abs().max().item()
    print(f"[option2] flash-linear vs naive ref:  rel_err={rel:.3e}  "
          f"{'OK' if rel < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    else:
        print("Run with --check on a CUDA GPU, or import gla_flash_triton.")
