"""
Standalone Stage 2 memory diagnostic.

Loads the Stage 2 model setup (base + Lizard attention + Stage 1 checkpoint + LoRA),
runs ONE forward + backward at the actual training shapes, and reports peak memory.

Use this to validate that Stage 2 will fit on your GPU before kicking off a 12+ hour
training run. Tweak USE_GRADIENT_CHECKPOINTING and DIAG_SEQ_LEN to find a config that fits.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import *
from lizard_attention import LizardAttention
from distill_lizard_llama_3_2_1B import (
    swap_attention,
    load_trainable,
    freeze_base_keep_lizard,
    is_lizard_param,
)


# ============================================================
# DIAGNOSTIC CONFIGURATION — tweak these to find what fits
# ============================================================
USE_GRADIENT_CHECKPOINTING = True
DIAG_SEQ_LEN = SEQ_LEN          # override here if you want to test a smaller value
DIAG_MICRO_BATCH = MICRO_BATCH  # usually 1
WARMUP_STEPS = 3                # run a few steps to get steady-state peak, not first-step peak


def maybe_enable_gradient_checkpointing(model, enable: bool, stage_name: str):
    """Enable gradient checkpointing if requested. Order matters with peft."""
    if not enable:
        print(f"  [{stage_name}] gradient checkpointing: OFF")
        return model

    # enable_input_require_grads MUST come before gradient_checkpointing_enable
    # when peft-wrapped models are involved
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"  [{stage_name}] gradient checkpointing: ON (use_reentrant=False)")
    return model


def gb(bytes_):
    return bytes_ / (1024 ** 3)


def print_memory_snapshot(label: str):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"  [{label}]")
    print(f"    allocated:    {gb(allocated):6.2f} GB")
    print(f"    reserved:     {gb(reserved):6.2f} GB")
    print(f"    peak so far:  {gb(peak):6.2f} GB")


def main():
    print("=" * 60)
    print("STAGE 2 MEMORY DIAGNOSTIC")
    print("=" * 60)
    print(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  USE_GRADIENT_CHECKPOINTING = {USE_GRADIENT_CHECKPOINTING}")
    print(f"  SEQ_LEN                    = {DIAG_SEQ_LEN}")
    print(f"  MICRO_BATCH                = {DIAG_MICRO_BATCH}")
    print(f"  LORA_RANK                  = {LORA_RANK}")
    print(f"  WINDOW_SIZE                = {WINDOW_SIZE}")
    print(f"  FEATURE_DIM                = {FEATURE_DIM}")
    print(f"  DTYPE                      = {DTYPE}")

    gpu_props = torch.cuda.get_device_properties(0)
    total_mem_gb = gpu_props.total_memory / (1024 ** 3)
    print(f"  GPU                        = {gpu_props.name} ({total_mem_gb:.1f} GB)")
    print()

    # ---- Build the Stage 2 model exactly as the training script does ----
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    print_memory_snapshot("after base model load")

    print("Swapping attention to Lizard...")
    model = swap_attention(model)
    print_memory_snapshot("after attention swap")

    print(f"Loading Stage 1 checkpoint from {STAGE1_CKPT}...")
    model = load_trainable(model, STAGE1_CKPT)
    model = freeze_base_keep_lizard(model)
    print_memory_snapshot("after stage1 checkpoint load")

    print("Attaching LoRA...")
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    for name, p in model.named_parameters():
        if is_lizard_param(name):
            p.requires_grad = True

    model = maybe_enable_gradient_checkpointing(
        model, USE_GRADIENT_CHECKPOINTING, "diag"
    )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train:,}  /  total: {n_total:,}  ({100 * n_train / n_total:.3f}%)")
    print_memory_snapshot("after model fully built")
    print()

    # ---- Build an optimizer; its state contributes to memory too ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=STAGE2_LR, betas=ADAM_BETAS, eps=ADAM_EPS)
    print_memory_snapshot("after optimizer construction")
    print()

    # ---- Run a few forward+backward+step cycles ----
    # First step has unrepresentative memory because optimizer state isn't
    # allocated yet (AdamW allocates m and v on the first .step() call).
    # Run >=2 steps to see steady-state peak.
    model.train()
    torch.cuda.reset_peak_memory_stats()

    test_input = torch.randint(0, 32000, (DIAG_MICRO_BATCH, DIAG_SEQ_LEN), device=DEVICE)

    for step in range(WARMUP_STEPS):
        print(f"Step {step + 1}/{WARMUP_STEPS}: forward + backward + optimizer step...")
        optim.zero_grad()

        out = model(input_ids=test_input, labels=test_input, use_cache=False)
        loss = out.loss

        loss.backward()
        optim.step()

        torch.cuda.synchronize()
        print(f"    loss = {loss.item():.4f}")
        print_memory_snapshot(f"end of step {step + 1}")
        print()

    # ---- Final summary ----
    final_peak = torch.cuda.max_memory_allocated()
    headroom = total_mem_gb - gb(final_peak)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Peak memory during training step: {gb(final_peak):6.2f} GB")
    print(f"  GPU total:                        {total_mem_gb:6.2f} GB")
    print(f"  Headroom:                         {headroom:+6.2f} GB")
    print()

    if headroom < 0:
        print("  OOM — peak exceeds GPU capacity. Will fail mid-training.")
        suggest_fixes(gb(final_peak), total_mem_gb)
    elif headroom < 2.0:
        print("  TIGHT — under 2 GB headroom. Memory fragmentation may cause")
        print("     OOM later in training even though this diagnostic passed.")
        suggest_fixes(gb(final_peak), total_mem_gb)
    else:
        print("  FITS — comfortable headroom, training should run cleanly.")
    print("=" * 60)


def suggest_fixes(peak_gb, total_gb):
    over_by = peak_gb - (total_gb - 2.0)
    print()
    print(f"  Need to save approximately {over_by:.1f} GB.")
    print()
    print("  Options ranked by impact:")
    print("    1. Enable gradient checkpointing (USE_GRADIENT_CHECKPOINTING = True)")
    print("       Expected savings: 5-10 GB")
    print("    2. Drop DIAG_SEQ_LEN to 1024 (and double NUM_EPOCHS to preserve tokens)")
    print("       Expected savings: 4-6 GB")
    print("    3. Switch AWA to chunked variant (see lizard_attention chunked version)")
    print("       Expected savings: 2-4 GB")
    print("    4. Drop LORA_RANK to 4")
    print("       Expected savings: <1 GB")


if __name__ == "__main__":
    main()