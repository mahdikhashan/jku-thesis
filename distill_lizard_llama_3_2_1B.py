"""
Lizard linearization for Llama-3.2-1B.

Single-script reproduction of both training stages from arXiv:2507.09025.
  Stage 1: MSE distillation of softmax attention outputs into Lizard attention.
  Stage 2: LoRA + Lizard params fine-tune with standard language-modeling loss.

Requires CUDA GPU with >=16 GB VRAM. Uses transformers, peft, datasets, wandb.

Stage 1 trains each LizardAttention layer-by-layer on the teacher's own
(input, output) pairs to avoid hidden-state drift across layers. The student
model is never run end-to-end during Stage 1; only its self_attn modules are.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb

from config import *

from lizard_attention import LizardAttention


# ============================================================
# MEMORY KNOBS
# ============================================================
# Toggle gradient checkpointing per stage. Stage 2 typically needs it on L4
# with the pure-PyTorch AWA at SEQ_LEN=2048. Stage 1 trains per-layer so
# memory pressure is lower; turn on only if Stage 1 also OOMs.

def maybe_enable_gradient_checkpointing(model, enable: bool, stage_name: str):
    """Enable gradient checkpointing if requested. Order matters with peft."""
    if not enable:
        print(f"  [{stage_name}] gradient checkpointing: OFF")
        return model

    # enable_input_require_grads MUST come before gradient_checkpointing_enable
    # when peft-wrapped models are involved (peft freezes base params; without
    # this call, gradients can't flow through frozen embeddings during recompute)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # use_reentrant=False is required for peft compatibility and avoids
    # in-place modification issues with the autograd graph
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print(f"  [{stage_name}] gradient checkpointing: ON (use_reentrant=False)")
    return model


# ============================================================
# MODEL SURGERY HELPERS
# ============================================================
def swap_attention(model):
    """Replace every LlamaAttention with LizardAttention; copy projection weights."""
    for layer_idx, layer in enumerate(model.model.layers):
        orig = layer.self_attn
        device = orig.q_proj.weight.device
        dtype = orig.q_proj.weight.dtype
        lizard = LizardAttention(model.config, layer_idx).to(device=device, dtype=dtype)
        lizard.q_proj.weight.data.copy_(orig.q_proj.weight.data)
        lizard.k_proj.weight.data.copy_(orig.k_proj.weight.data)
        lizard.v_proj.weight.data.copy_(orig.v_proj.weight.data)
        lizard.o_proj.weight.data.copy_(orig.o_proj.weight.data)
        layer.self_attn = lizard
    return model


def is_lizard_param(name: str) -> bool:
    return any(k in name for k in LIZARD_PARAM_KEYS)


def freeze_base_keep_lizard(model):
    """Freeze everything except Lizard-added parameters."""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if is_lizard_param(name):
            p.requires_grad = True
    return model


def save_trainable(model, path: Path):
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sd, path)


def load_trainable(model, path: Path):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys in checkpoint: {unexpected[:3]}...")
    return model


# ============================================================
# DATA
# ============================================================
class PackedDataset(Dataset):
    def __init__(self, ids: torch.Tensor):
        self.ids = ids  # (N, SEQ_LEN)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.ids[i]
        return {"input_ids": x, "labels": x.clone()}


def build_dataloader(tokenizer) -> DataLoader:
    """Tokenize cleaned Alpaca and pack into SEQ_LEN chunks."""
    raw = (
        load_dataset(DATASET_NAME, split="train")
        .shuffle(seed=SEED)
        .select(range(DATASET_SUBSET))
    )

    def format_example(ex):
        if ex.get("input"):
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}{tokenizer.eos_token}"
            )
        else:
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}{tokenizer.eos_token}"
            )
        return {"text": text}

    raw = raw.map(format_example, remove_columns=raw.column_names)

    def tok_fn(batch):
        return tokenizer(batch["text"], add_special_tokens=False, truncation=False)

    tokenized = raw.map(tok_fn, batched=True, remove_columns=["text"])

    # Concatenate and chunk
    all_ids = []
    for row in tokenized:
        all_ids.extend(row["input_ids"])
    n_chunks = len(all_ids) // SEQ_LEN
    all_ids = all_ids[: n_chunks * SEQ_LEN]
    chunks = torch.tensor(all_ids, dtype=torch.long).view(n_chunks, SEQ_LEN)
    print(f"  packed {n_chunks} sequences of {SEQ_LEN} tokens (~{n_chunks * SEQ_LEN / 1e6:.1f}M total)")

    return DataLoader(PackedDataset(chunks), batch_size=MICRO_BATCH, shuffle=True, drop_last=True)


# ============================================================
# STAGE 1: ATTENTION APPROXIMATION (MSE)
# ============================================================
def stage1_distill():
    print("=" * 60)
    print("STAGE 1: ATTENTION APPROXIMATION (MSE)")
    print("=" * 60)

    wandb.init(
        project=WANDB_PROJECT,
        name="stage1_distill",
        config={
            "stage": 1, "model": MODEL_NAME, "lr": STAGE1_LR, "epochs": NUM_EPOCHS,
            "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM,
            "window_size": WINDOW_SIZE, "meta_tokens": NUM_META_TOKENS,
            "grad_checkpointing": USE_GRADIENT_CHECKPOINTING_STAGE1,
        },
    )

    # ---- Teacher: frozen, all softmax ----
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Student: same weights, attention swapped to Lizard ----
    student = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    student = swap_attention(student)
    student = freeze_base_keep_lizard(student)

    # Stage 1 runs layers individually (no full forward), so gradient
    # checkpointing on the wrapper model is mostly a no-op here. Provided as
    # a flag for symmetry; safe to leave off unless Stage 1 itself OOMs.
    student = maybe_enable_gradient_checkpointing(
        student, USE_GRADIENT_CHECKPOINTING_STAGE1, "stage1"
    )

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"  trainable: {n_train:,}  /  total: {n_total:,}  ({100 * n_train / n_total:.3f}%)")

    # ---- Hook teacher: capture self_attn (input, output) per layer ----
    teacher_inputs = {}
    teacher_outputs = {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            teacher_inputs[idx] = x.detach()
            teacher_outputs[idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook

    for i, layer in enumerate(teacher.model.layers):
        layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)

    # ---- Tokenizer + data ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    loader = build_dataloader(tokenizer)

    # ---- Optimizer + scheduler ----
    trainable = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=STAGE1_LR, betas=ADAM_BETAS, eps=ADAM_EPS)
    total_steps = (len(loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"  total optimizer steps: {total_steps}  (warmup: {warmup_steps})")

    n_layers = len(student.model.layers)
    step = 0
    optim.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)

            teacher_inputs.clear()
            teacher_outputs.clear()

            # Teacher forward — populates hook dicts
            with torch.no_grad():
                teacher(input_ids=input_ids, use_cache=False)

            # Per-layer MSE; backward after each layer to keep memory flat
            total_loss = 0.0
            for i, layer in enumerate(student.model.layers):
                x = teacher_inputs[i]
                y_target = teacher_outputs[i]
                y_pred, _ = layer.self_attn(x)
                layer_loss = F.mse_loss(y_pred.float(), y_target.float()) / n_layers
                (layer_loss / GRAD_ACCUM).backward()
                total_loss += layer_loss.item()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if step % LOG_EVERY == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"[stage1] epoch {epoch} step {step}/{total_steps} loss {total_loss:.5f} lr {lr:.2e}")
                    wandb.log({"stage1/loss": total_loss, "stage1/lr": lr, "stage1/step": step})

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_trainable(student, STAGE1_CKPT)
    print(f"  Stage 1 checkpoint -> {STAGE1_CKPT}")
    wandb.finish()

    # Free teacher
    del teacher
    torch.cuda.empty_cache()


# ============================================================
# STAGE 2: LANGUAGE-MODELING FINE-TUNE (LoRA + Lizard params)
# ============================================================
def stage2_finetune():
    print("=" * 60)
    print("STAGE 2: LANGUAGE-MODELING FINE-TUNE")
    print("=" * 60)

    wandb.init(
        project=WANDB_PROJECT,
        name="stage2_finetune",
        config={
            "stage": 2, "model": MODEL_NAME, "lr": STAGE2_LR, "epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_targets": LORA_TARGETS,
            "grad_checkpointing": USE_GRADIENT_CHECKPOINTING_STAGE2,
        },
    )

    # Load base, swap attention, load Stage 1 weights, freeze base
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model = swap_attention(model)
    model = load_trainable(model, STAGE1_CKPT)
    model = freeze_base_keep_lizard(model)

    # Attach LoRA to q/k/v of every attention layer
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # peft may freeze everything except adapters; re-enable Lizard params
    for name, p in model.named_parameters():
        if is_lizard_param(name):
            p.requires_grad = True

    # Gradient checkpointing — must come AFTER peft wrapping. The helper handles
    # the correct ordering: enable_input_require_grads -> gradient_checkpointing_enable
    # with use_reentrant=False (required for peft compatibility).
    model = maybe_enable_gradient_checkpointing(
        model, USE_GRADIENT_CHECKPOINTING_STAGE2, "stage2"
    )

    # VERIFY GC IS ACTUALLY ON
    print(f"  model.is_gradient_checkpointing: {model.is_gradient_checkpointing}")
    if hasattr(model, "base_model"):
        print(f"  base_model.is_gradient_checkpointing: {model.base_model.is_gradient_checkpointing}")
        if hasattr(model.base_model, "model"):
            inner = model.base_model.model
            print(f"  base_model.model.is_gradient_checkpointing: {inner.is_gradient_checkpointing}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {n_train:,}  /  total: {n_total:,}  ({100 * n_train / n_total:.3f}%)")

    # Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    loader = build_dataloader(tokenizer)

    # Optimizer + scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=STAGE2_LR, betas=ADAM_BETAS, eps=ADAM_EPS)

    # DIAGNOSTIC
    print("\n=== Parameters in optimizer ===")
    lora_count = 0
    lizard_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if 'lora' in name.lower():
                lora_count += 1
            elif any(k in name for k in ('meta_tokens', 'alpha_blend', 'phi_q', 'phi_k', 'W_gamma')):
                lizard_count += 1
                print(f"  LIZARD: {name}  shape={tuple(p.shape)}")
            else:
                print(f"  OTHER: {name}  shape={tuple(p.shape)}")
    print(f"LoRA params: {lora_count}, Lizard params: {lizard_count}")
    print(f"Total trainable: {sum(p.numel() for p in trainable):,}")

    total_steps = (len(loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    print(f"  total optimizer steps: {total_steps}  (warmup: {warmup_steps})")

    step = 0
    optim.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = out.loss

            (loss / GRAD_ACCUM).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if step % LOG_EVERY == 0:
                    lr = sched.get_last_lr()[0]
                    print(f"[stage2] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {lr:.2e}")
                    wandb.log({"stage2/loss": loss.item(), "stage2/lr": lr, "stage2/step": step})

    # Merge LoRA and save full state dict (LizardAttention is not a HF registered
    # architecture, so save_pretrained alone would not be reloadable correctly;
    # save the full state dict instead, and reconstruct via swap_attention + load).
    # === Before merge_and_unload, check Lizard param values ===
    print("\n=== Lizard params BEFORE merge_and_unload ===")
    for i in [0, 5, 10, 15]:
        attn = model.base_model.model.model.layers[i].self_attn
        print(f"Layer {i}: alpha={attn.alpha_blend.item():.4f}, "
              f"meta={[f'{m:.4f}' for m in attn.meta_tokens]}")

    model = model.merge_and_unload()

    print("\n=== Lizard params AFTER merge_and_unload ===")
    for i in [0, 5, 10, 15]:
        attn = model.model.layers[i].self_attn  # no more base_model wrapper
        print(f"Layer {i}: alpha={attn.alpha_blend.item():.4f}, "
              f"meta={[f'{m:.4f}' for m in attn.meta_tokens]}")

    full_sd = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(full_sd, STAGE2_CKPT)
    print(f"  Stage 2 full state dict -> {STAGE2_CKPT}")
    print(f"  To reload: load base Llama-3.2-1B, call swap_attention(m), then m.load_state_dict(torch.load(STAGE2_CKPT), strict=False)")
    wandb.finish()


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # stage1_distill()
    stage2_finetune()
    print("Done.")


if __name__ == "__main__":
    main()