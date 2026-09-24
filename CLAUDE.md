# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis codebase implementing "Lizard" linearization (arXiv:2507.09025) applied to Llama-3.2-1B: replacing full softmax attention with a hybrid of Gated Linear Attention (GLA) and a windowed softmax branch (AWA), trained via two-stage distillation from the original model. There is no package/build system — everything is run as standalone Python scripts against conda environments defined per-task in `envs/`.

## Environments

There is no single environment; pick the conda env for the task from `envs/*.yaml` and create it before running anything:

```bash
conda env create -f envs/distill_lizard_llama_3_2_1B.yaml   # training/distillation (torch+cu128, transformers, peft, wandb, lm-eval)
conda env create -f envs/gla_kernel_benchmark.yaml            # Triton/FLA kernel benchmarking (torch 2.7, cuda 12.4, flash-linear-attention)
conda env create -f envs/lm-eval-llama-3-2-1B-teacher.yaml    # baseline teacher eval only (torch+cu121, lm-eval)
```

Recent commit history shows the `gla_kernel_benchmark` env is GPU-sensitive (A10/A100/P40/L4 all needed different hacks for `fla`/Triton compatibility — see `hack fla to fix ...` and `modify for A10`/`modify for tesla P40` commits). If kernel code breaks after a GPU change, check those commits before re-deriving a fix.

## Commands

No test runner, linter, or build tool is configured — run scripts directly with `python`.

```bash
# Full two-stage distillation (stage1: attention MSE distill, stage2: LoRA + Lizard fine-tune)
python distill_lizard_llama_3_2_1B.py

# Generate from the trained Lizard checkpoint
python generate.py

# Side-by-side teacher vs. Lizard generation
python generate_teacher.py

# lm-eval-harness benchmarks (piqa, arc_easy, arc_challenge, hellaswag, winogrande)
python eval.py

# Kernel correctness check + benchmark (each kernels/*.py supports --check)
python kernels/gla_triton_flash.py --check
python kernels/gla_triton_benchmark.py
python kernels/gla_torch_benchmark.py

# Numerical correctness / diagnostic scripts (the closest thing to a test suite)
python debug/gla_cosine_similarity.py            # validates gla_branch against a slow reference (expect cosine ~0.999)
python debug/awa_numerical_test.py               # validates awa_branch (windowed softmax + meta-token denominator)
python debug/per_layer_cosine_similarity.py      # per-layer teacher-vs-Lizard drift after training
python debug/stage_2_memory_diagnostic.py        # one forward+backward at training shapes; checks peak memory before a full run
```

All hyperparameters, paths, and stage configs live in [config.py](config.py) — edit there rather than passing CLI flags (scripts have none).

## Architecture

**Data flow:** `config.py` (all constants, via `from config import *`) → `lizard_attention.py` (the `LizardAttention` module) → `distill_lizard_llama_3_2_1B.py` (swaps attention into a HF model and runs the two training stages) → checkpoints in `checkpoints/` → consumed by `eval.py` / `generate.py` / `generate_teacher.py` / `debug/*.py`.

**`LizardAttention` (`lizard_attention.py`)** is a drop-in replacement for `LlamaAttention`: `output = GLA(x) + alpha_blend * AWA(x)`, with RoPE removed entirely.
- `gla_branch`: gated linear attention via FLA's `fused_recurrent_gla`, run twice (once with `v`, once with `v=ones`) to get a normalized numerator/denominator — this is the *global* memory path.
- `awa_branch`: hand-chunked sliding-window softmax attention (never materializes the full L×L matrix) with extra "meta-token" logits acting as denominator-only attention sinks — this is the *local* precision path.
- New learnable parameters introduced by Lizard: `phi_q`, `phi_k` (Hedgehog feature maps), `W_gamma` (per-token scalar gate), `meta_tokens`, `alpha_blend`. Their names are enumerated in `config.LIZARD_PARAM_KEYS` and that list is the single source of truth for "is this a Lizard param" (`is_lizard_param`) used throughout training, freezing, and checkpoint save/load.
- Mixed precision is load-bearing, not incidental: Lizard params are upcast to fp32 (bf16 resolution is too coarse for their AdamW updates) while the base model stays bf16. Every branch has explicit `=== DTYPE BOUNDARY ===` comments where fp32/bf16 tensors meet — preserve these casts when editing; removing them silently breaks either correctness or Tensor Core usage.
- `use_cache=False` is required everywhere Lizard models run generation — Lizard doesn't support KV caching.

**Two-stage training (`distill_lizard_llama_3_2_1B.py`):**
1. **Stage 1** (`stage1_distill`): freeze everything except Lizard params, register forward hooks on the frozen teacher's attention layers, and MSE-distill each student `LizardAttention` layer independently against the teacher's per-layer (input, output) pair — layers are trained independently per batch (per-layer backward, not a summed loss) to keep peak memory flat. Produces `STAGE1_CKPT` (Lizard params only).
2. **Stage 2** (`stage2_finetune`): load the Stage 1 checkpoint, wrap in a LoRA adapter (`LORA_TARGETS = q/k/v_proj`), re-unfreeze + re-upcast Lizard params (PEFT wrapping resets requires_grad/dtype), then standard causal-LM fine-tuning on the same Alpaca data. Ends with `merge_and_unload()` and saves a full state dict to `STAGE2_CKPT`.
3. Reload pattern for any downstream script: load base Llama-3.2-1B → `swap_attention(model)` → `load_state_dict(torch.load(STAGE2_CKPT), strict=False)`.

**`kernels/`** is an independent line of work from the training pipeline: it explores hardware-aware implementations of Lizard's log-space GLA reparameterization (the paper's trick for avoiding bf16 underflow in the cumulative gate product), progressing CPU prototype → Triton fused-score kernel → full Triton flash-style kernel, each benchmarked against FLA's own `chunk_gla`. These files are self-contained and not imported by the training script.

**`debug/`** scripts are one-off numerical/diagnostic checks (cosine similarity vs. a slow reference implementation, memory profiling, checkpoint key inspection) written while debugging distillation quality — treat them as the test suite in absence of a real one, and follow their pattern (compare against a slow/naive reference, assert high cosine similarity) when adding new correctness checks.
