"""
Unified GLA benchmark — run on your L4 / A40.

Times the Lizard reparameterization across implementations:

  torch_reparam        gla_lizard_reparam (exp/cat + torch matmuls; cuBLAS)
  triton_fused_scores  Option 1: reparam fused into the score GEMM (Triton),
                       scores@V left to torch
  triton_flash         Option 2: full flash-style fused kernel, no L x L matrix

All compute the same numerator Y = (tril(Q~ K~^T)) V. Validate correctness
first with each file's --check; then run this for latency.

Usage:
    python bench_gla_triton.py
    python bench_gla_triton.py --L 8192
"""

import argparse
import time
import torch


def _torch_reparam(x_q, x_k, v, gamma, W):
    L = x_q.shape[2]
    QW = x_q @ W
    KW = x_k @ W
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1).unsqueeze(-1)
    Q = torch.cat([torch.exp(QW + logC), torch.exp(-QW + logC)], dim=-1)
    K = torch.cat([torch.exp(KW - logC), torch.exp(-KW - logC)], dim=-1)
    s = Q @ K.transpose(-2, -1)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=v.device))
    s = s.masked_fill(~causal, 0.0)
    return s @ v


def timed(fn, *a, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*a)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = "cuda"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    B, H, L, d, F = args.B, args.H, args.L, args.d, args.F
    Vd = d
    print(f"shapes B={B} H={H} L={L} d={d} F={F} Vd={Vd}  dtype={args.dtype}")
    print(f"device {torch.cuda.get_device_name()}")
    print("-" * 56)

    x_q = torch.randn(B, H, L, d, device=dev, dtype=dt)
    x_k = torch.randn(B, H, L, d, device=dev, dtype=dt)
    v = torch.randn(B, H, L, Vd, device=dev, dtype=dt)
    W = (torch.randn(d, F, device=dev) * 0.3).to(dt)
    gamma = (torch.sigmoid(torch.randn(B, H, L, device=dev)) * 0.1 + 0.9).to(dt)

    res = {}
    res["torch_reparam"] = timed(_torch_reparam, x_q, x_k, v, gamma, W)

    try:
        from gla_triton_fused_scores import gla_fused_scores
        res["triton_fused_scores"] = timed(gla_fused_scores, x_q, x_k, v, gamma, W)
    except Exception as e:
        print(f"[skip] option1: {e}")

    try:
        from gla_triton_flash import gla_flash_triton
        res["triton_flash"] = timed(gla_flash_triton, x_q, x_k, v, gamma, W)
    except Exception as e:
        print(f"[skip] option2: {e}")

    # FLA standard GLA — the paper's baseline (Yang & Zhang, 2024).
    # Apples-to-apples: same hedgehog feature map phi(xW) = [exp(xW) ⊕ exp(-xW)]
    # and same scalar gate, but the gate is passed SEPARATELY to FLA's chunked
    # kernel (gk = log gamma, broadcast across the 2F feature dim) rather than
    # folded into q/k. This is exactly the "standard GLA" the reparam replaces.
    try:
        from fla.ops.gla import chunk_gla, fused_recurrent_gla

        def _phi(xW):
            return torch.cat([torch.exp(xW), torch.exp(-xW)], dim=-1)

        # Precompute feature maps once (the reparam paths also do their projection
        # inside the timed call, so to stay fair we include phi+projection here too).
        def _fla_chunk():
            phi_q = _phi(x_q @ W)                         # (B,H,L,2F)
            phi_k = _phi(x_k @ W)
            # FLA expects (B, L, H, D); gk is the per-step log-decay, shape (B,L,H,2F)
            q_f = phi_q.transpose(1, 2).contiguous()
            k_f = phi_k.transpose(1, 2).contiguous()
            v_f = v.transpose(1, 2).contiguous()
            gk = torch.log(gamma.clamp(min=1e-12))        # (B,H,L)
            gk = gk[..., None].expand(B, H, L, phi_q.shape[-1])
            gk = gk.transpose(1, 2).contiguous()
            return chunk_gla(q_f, k_f, v_f, gk)

        res["fla_chunk_gla"] = timed(_fla_chunk)
    except Exception as e:
        print(f"[skip] fla: {e}")

    base = res["torch_reparam"]
    print(f"{'impl':<22}{'ms':>10}{'vs torch':>12}{'vs FLA':>12}")
    fla = res.get("fla_chunk_gla")
    for name, ms in res.items():
        vs_torch = f"{base / ms:.2f}x"
        vs_fla = f"{fla / ms:.2f}x" if fla else "--"
        print(f"{name:<22}{ms:>10.3f}{vs_torch:>12}{vs_fla:>12}")


if __name__ == "__main__":
    main()
