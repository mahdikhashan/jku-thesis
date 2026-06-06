"""
Reproduce Lizard Table 3: forward-pass latency, standard parallel GLA vs the
Lizard log-space reparameterization, in bf16, on your GPU.

The paper's claim (A100-80GB): Lizard is +32% to +36% faster than standard GLA
because the reparameterization keeps the whole op in bf16 on Tensor Cores,
whereas the standard parallel form's cumulative-product C underflows in bf16 and
forces an fp32 fallback.

This script times two forward passes at the paper's (B, L) configurations:

  standard_parallel_fp32   the unstable parallel form, run in fp32 (the fallback
                           the paper says standard GLA is forced into for bf16)
  lizard_reparam_bf16      the log-space GEMM form, run in bf16 on Tensor Cores

and reports the speedup, to compare against the paper's +32-36%.

NOTE
  - Needs a CUDA GPU. The +32-36% figure is A100-specific (strong Tensor Cores);
    on L4 / A40 the gap will differ but the *direction* (reparam faster, and
    able to run bf16 without NaNs) is the reproducible claim.
  - Run gla_lizard_reparam.py first (CPU) to confirm correctness of the math.
"""

import argparse
import time
import torch


def hedgehog(xW):
    return torch.cat([torch.exp(xW), torch.exp(-xW)], dim=-1)


def standard_parallel(x_q, x_k, v, gamma, W):
    """Standard parallel GLA: Y = ((phi(Q)⊙C)(phi(K)/C)^T ⊙ M) V.

    Uses the raw cumulative product C, which is why it needs fp32. Call this
    with fp32 tensors; in bf16 the 1/C term overflows to inf/NaN.
    """
    L = x_q.shape[2]
    phi_q = hedgehog(x_q @ W)
    phi_k = hedgehog(x_k @ W)
    C = torch.cumprod(gamma.clamp(min=1e-12), dim=-1).unsqueeze(-1)  # (B,H,L,1)
    q = phi_q * C
    k = phi_k / C
    scores = q @ k.transpose(-2, -1)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=v.device))
    scores = scores.masked_fill(~causal, 0.0)
    return scores @ v


def lizard_reparam(x_q, x_k, v, gamma, W):
    """Lizard log-space GEMM form. Safe in bf16."""
    L = x_q.shape[2]
    QW = x_q @ W
    KW = x_k @ W
    logC = torch.cumsum(torch.log(gamma.clamp(min=1e-12)), dim=-1).unsqueeze(-1)
    Q = torch.cat([torch.exp(QW + logC), torch.exp(-QW + logC)], dim=-1)
    K = torch.cat([torch.exp(KW - logC), torch.exp(-KW - logC)], dim=-1)
    scores = Q @ K.transpose(-2, -1)
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=v.device))
    scores = scores.masked_fill(~causal, 0.0)
    return scores @ v


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
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--d", type=int, default=64)     # head_dim
    ap.add_argument("--F", type=int, default=128)    # feature_dim
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    dev = "cuda"
    print(f"device {torch.cuda.get_device_name()}   H={args.H} d={args.d} F={args.F}")
    print(f"{'config':<22}{'standard fp32':>16}{'lizard bf16':>14}{'speedup':>10}")
    print("-" * 62)

    # paper Table 3 configurations
    configs = [(16, 2048), (16, 4096), (16, 8192), (32, 8192)]
    for B, L in configs:
        W = (torch.randn(args.d, args.F, device=dev) * 0.5)
        x_q = torch.randn(B, args.H, L, args.d, device=dev)
        x_k = torch.randn(B, args.H, L, args.d, device=dev)
        v = torch.randn(B, args.H, L, args.d, device=dev)
        gamma = torch.sigmoid(torch.randn(B, args.H, L, device=dev)) * 0.1 + 0.9

        # standard in fp32 (the forced fallback)
        ms_std = timed(standard_parallel, x_q.float(), x_k.float(), v.float(),
                       gamma.float(), W.float(), iters=10)

        # lizard in bf16 on Tensor Cores
        ms_liz = timed(lizard_reparam,
                       x_q.bfloat16(), x_k.bfloat16(), v.bfloat16(),
                       gamma.bfloat16(), W.bfloat16(), iters=10)

        sp = (ms_std - ms_liz) / ms_std * 100
        print(f"B={B:<3} L={L:<6}        {ms_std:>12.2f}ms{ms_liz:>11.2f}ms{sp:>+8.0f}%")


if __name__ == "__main__":
    main()
