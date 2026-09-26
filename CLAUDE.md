# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis codebase implementing "Lizard" linearization (arXiv:2507.09025) applied to Llama-3.2-1B: replacing softmax attention with gated linear attention (GLA) plus a sliding-window softmax branch with sink tokens (AWA), trained via two-stage distillation from the original model. No package/build system — standalone scripts run inside the conda envs in `envs/`. Setup and run steps for humans are in [README.md](README.md).

## Commands

```bash
conda activate distill_lizard_llama_3_2_1B   # env for everything except kernels/

pytest                                                  # full suite, CPU-only, tiny models
pytest test_lizard.py::test_gla_matches_formula_loops   # single test
LIZARD_TEST_MODEL=meta-llama/Llama-3.2-1B pytest test_lizard.py::test_whole_model_logits_unchanged_after_swap

python train.py              # stage 1 + stage 2, needs CUDA; WANDB_MODE=disabled to skip wandb
python evaluate.py           # lm-eval on the trained Lizard checkpoint
python evaluate.py llama     # same tasks on the unmodified teacher
python generate.py           # greedy answers from teacher then Lizard; pass `llama` or `lizard` for one
python benchmark.py          # accuracy (and GPU speed) vs FLA / SDPA / gpt-oss baselines
```

All hyperparameters and paths live in [config.py](config.py); scripts take no flags (except `evaluate.py`'s model choice). Tests shrink config values via `monkeypatch.setattr(C, ...)`, so code must read config as `C.NAME` at call time, not `from config import NAME`.

## Architecture

- **`lizard_attention.py`** — pure-PyTorch, dense (materializes L×L) reference implementation. `hedgehog`, `gla`, `awa` are free functions so tests and baselines can call them directly; `LizardAttention` is a drop-in for `LlamaAttention` (returns `(out, None)`, no RoPE, no KV cache so generation needs `use_cache=False`). `from_llama` builds one from a Llama attention layer by copying q/k/v/o weights.
- **`reference.py`** — slow loop/recurrent versions of every formula (`gla_loop`, `gla_recurrent`, `awa_loop`, `lizard_loop`) plus tiny-config and random-input helpers. It is the ground truth; tests assert `rel_err < 1e-10` in float64 against it.
- **`baselines.py`** — adapters that reshape inputs to call external implementations (FLA simple_gla / parallel_attn, SDPA with a sink column, HF and OpenAI gpt-oss sink attention). Multiple meta tokens collapse to one sink via `logsumexp`. Used by `test_baselines.py` and `benchmark.py`; optional deps are imported lazily and tests `importorskip` them.
- **`train.py`**:
  - Stage 1 builds standalone student `LizardAttention` layers (not inserted into the model), captures each teacher attention layer's (input, output) with forward hooks, and trains only Lizard params with a per-layer summed-squared-error loss, backpropagating layer by layer. Returns a state dict keyed like `model.layers.{i}.self_attn.*`.
  - Stage 2 swaps Lizard into a fresh model (`to_lizard`), loads the stage 1 params, wraps with LoRA on q/k/v, re-enables Lizard params (PEFT freezes them), trains with causal-LM loss, then `merge_and_unload()` so the saved checkpoint loads with `strict=True`.
  - `is_lizard` matches parameter names by exact dotted component against `C.LIZARD_PARAMS` — the single source of truth for which params are Lizard-owned.
- **`evaluate.py`** — wraps the model in lm-eval's `HFLM`; loads Lizard weights with `strict=True`. Its `load(which)` is the shared teacher/Lizard loader that `generate.py` also uses.
- **`debug/`** — legacy diagnostic scripts written for the old training code (cosine similarity vs. teacher, memory checks). They import the removed `distill_lizard_llama_3_2_1B.py` and old config names, so they do not run against the current code without porting.
- **`notebooks/`** — teacher lm-eval notebook and its saved results (`envs/lm-eval-llama-3-2-1B-teacher.yaml`).
- **`kernels/`** — independent Triton experiments on Lizard's log-space GLA reparameterization, with their own env (`envs/gla_kernel_benchmark.yaml`). Not imported by the rest of the code. That env is GPU-sensitive; past fixes for A10/P40/L4 are in git history (`hack fla ...`, `modify for A10` commits).
