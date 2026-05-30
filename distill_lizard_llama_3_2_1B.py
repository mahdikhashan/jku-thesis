"""
Lizard linearization for Llama-3.2-1B (arXiv:2507.09025).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from pathlib import Path

import torch
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


logger = logging.getLogger("lizard")


# ------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------
def enable_gradient_checkpointing(model, enable: bool, stage: str):
    """Turn on gradient checkpointing. Must run after peft wrapping in stage 2."""
    if not enable:
        logger.info("[%s] gradient checkpointing: off", stage)
        return model

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    logger.info("[%s] gradient checkpointing: on", stage)
    return model


def upcast_lizard_params_to_fp32(model) -> int:
    """Cast trainable Lizard parameters to fp32. Returns how many were changed.

    bf16's resolution near unit magnitude (~7.8e-3) is coarser than a typical
    AdamW step (~5e-4), so updates to small bf16 params silently round away.
    fp32 (~1.2e-7 resolution) makes every update land. Cost is ~1.5 MB including
    optimizer state.
    """
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and is_lizard_param(name) and p.dtype != torch.float32:
            p.data = p.data.float()
            count += 1
    return count


def swap_attention(model):
    """Replace each LlamaAttention with LizardAttention, copying projection weights."""
    for layer_idx, layer in enumerate(model.model.layers):
        orig = layer.self_attn
        lizard = LizardAttention(model.config, layer_idx).to(
            device=orig.q_proj.weight.device, dtype=orig.q_proj.weight.dtype
        )
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(lizard, proj).weight.data.copy_(getattr(orig, proj).weight.data)
        layer.self_attn = lizard
    return model


def is_lizard_param(name: str) -> bool:
    return any(k in name for k in LIZARD_PARAM_KEYS)


def freeze_base_keep_lizard(model):
    """Freeze every parameter except the Lizard-added ones."""
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
    _, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        logger.warning("unexpected keys when loading %s: %s ...", path, unexpected[:3])
    return model


def log_lizard_values(model, label: str, layers=(0, 5, 10, 15)):
    """Print per-layer alpha_blend and meta_tokens. Confirms params actually moved."""
    logger.info("=== %s ===", label)
    for i in layers:
        attn = model.model.layers[i].self_attn
        meta = [f"{m:.4f}" for m in attn.meta_tokens.detach().cpu()]
        logger.info("layer %2d: alpha=%.6f  meta=%s", i, attn.alpha_blend.item(), meta)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
class PackedDataset(Dataset):
    def __init__(self, ids: torch.Tensor):
        self.ids = ids

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
    tokenized = raw.map(
        lambda b: tokenizer(b["text"], add_special_tokens=False, truncation=False),
        batched=True,
        remove_columns=["text"],
    )

    all_ids = []
    for row in tokenized:
        all_ids.extend(row["input_ids"])

    n_chunks = len(all_ids) // SEQ_LEN
    all_ids = all_ids[: n_chunks * SEQ_LEN]
    chunks = torch.tensor(all_ids, dtype=torch.long).view(n_chunks, SEQ_LEN)

    logger.info("packed %d sequences of %d tokens (~%.1fM total)",
                n_chunks, SEQ_LEN, n_chunks * SEQ_LEN / 1e6)

    return DataLoader(
        PackedDataset(chunks), batch_size=MICRO_BATCH, shuffle=True, drop_last=True
    )


# ------------------------------------------------------------------
# Stage 1: attention distillation
# ------------------------------------------------------------------
def stage1_distill():
    logger.info("STAGE 1: attention distillation (MSE)")

    wandb.init(
        project=WANDB_PROJECT,
        name="stage1_distill",
        config={
            "stage": 1, "model": MODEL_NAME, "lr": STAGE1_LR, "epochs": NUM_EPOCHS,
            "seq_len": SEQ_LEN, "feature_dim": FEATURE_DIM,
            "window_size": WINDOW_SIZE, "meta_tokens": NUM_META_TOKENS,
        },
    )

    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    student = swap_attention(student)
    student = freeze_base_keep_lizard(student)

    n_upcast = upcast_lizard_params_to_fp32(student)
    logger.info("upcast %d Lizard parameters to fp32", n_upcast)

    student = enable_gradient_checkpointing(
        student, USE_GRADIENT_CHECKPOINTING_STAGE1, "stage1"
    )

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    logger.info("trainable %d / %d (%.3f%%)", n_train, n_total, 100 * n_train / n_total)

    # Capture each teacher attention layer's (input, output) via forward hooks.
    teacher_inputs, teacher_outputs = {}, {}

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            teacher_inputs[idx] = x.detach()
            out = output[0] if isinstance(output, tuple) else output
            teacher_outputs[idx] = out.detach()
        return hook

    for i, layer in enumerate(teacher.model.layers):
        layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    loader = build_dataloader(tokenizer)

    trainable = [p for p in student.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=STAGE1_LR, betas=ADAM_BETAS, eps=ADAM_EPS)
    total_steps = (len(loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    logger.info("optimizer steps: %d (warmup %d)", total_steps, warmup_steps)

    n_layers = len(student.model.layers)
    step = 0
    optim.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)

            teacher_inputs.clear()
            teacher_outputs.clear()
            with torch.no_grad():
                teacher(input_ids=input_ids, use_cache=False)

            # Train each attention module on the teacher's own layer input.
            # Per-element mean MSE averaged over layers. (The literal Frobenius
            # norm^2 from the paper trained worse here: its per-layer magnitude
            # imbalance let high-magnitude layers dominate the gradient. Mean MSE
            # normalizes per element and gave better attention matching and eval.)
            # Layers are independent, so per-layer backward equals a summed
            # backward but keeps peak memory flat.
            total_loss = 0.0
            for i, layer in enumerate(student.model.layers):
                y_pred, _ = layer.self_attn(teacher_inputs[i])
                layer_loss = F.mse_loss(y_pred.float(), teacher_outputs[i].float()) / n_layers
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
                    logger.info(
                        "[stage1] epoch %d step %d/%d loss %.2f lr %.2e",
                        epoch, step, total_steps, total_loss, lr,
                    )
                    wandb.log({"stage1/loss": total_loss, "stage1/lr": lr, "stage1/step": step})

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_trainable(student, STAGE1_CKPT)
    logger.info("stage 1 checkpoint -> %s", STAGE1_CKPT)
    log_lizard_values(student, "Stage 1 final Lizard values")

    wandb.finish()
    del teacher
    torch.cuda.empty_cache()


# ------------------------------------------------------------------
# Stage 2: LoRA + Lizard fine-tune
# ------------------------------------------------------------------
def stage2_finetune():
    logger.info("STAGE 2: language-modeling fine-tune (LoRA + Lizard)")

    wandb.init(
        project=WANDB_PROJECT,
        name="stage2_finetune",
        config={
            "stage": 2, "model": MODEL_NAME, "lr": STAGE2_LR, "epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_targets": LORA_TARGETS,
        },
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(DEVICE)
    model = swap_attention(model)
    model = load_trainable(model, STAGE1_CKPT)
    model = freeze_base_keep_lizard(model)
    upcast_lizard_params_to_fp32(model)

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # peft freezes non-adapter params; re-enable Lizard params and re-upcast,
    # since wrapping can reset their dtype.
    for name, p in model.named_parameters():
        if is_lizard_param(name):
            p.requires_grad = True
    upcast_lizard_params_to_fp32(model)

    model = enable_gradient_checkpointing(
        model, USE_GRADIENT_CHECKPOINTING_STAGE2, "stage2"
    )

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("trainable %d / %d (%.3f%%)", n_train, n_total, 100 * n_train / n_total)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    loader = build_dataloader(tokenizer)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=STAGE2_LR, betas=ADAM_BETAS, eps=ADAM_EPS)
    total_steps = (len(loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    logger.info("optimizer steps: %d (warmup %d)", total_steps, warmup_steps)

    step = 0
    optim.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            (out.loss / GRAD_ACCUM).backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if step % LOG_EVERY == 0:
                    lr = sched.get_last_lr()[0]
                    logger.info(
                        "[stage2] epoch %d step %d/%d loss %.4f lr %.2e",
                        epoch, step, total_steps, out.loss.item(), lr,
                    )
                    wandb.log({"stage2/loss": out.loss.item(), "stage2/lr": lr, "stage2/step": step})

    model = model.merge_and_unload()
    log_lizard_values(model, "Stage 2 final Lizard values (after merge)")

    full_sd = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(full_sd, STAGE2_CKPT)
    logger.info("stage 2 state dict -> %s", STAGE2_CKPT)
    logger.info(
        "reload: load base Llama-3.2-1B, swap_attention(m), "
        "m.load_state_dict(torch.load(STAGE2_CKPT), strict=False)"
    )
    wandb.finish()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("PYTORCH_CUDA_ALLOC_CONF = %s", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_distill()
    stage2_finetune()
    logger.info("done")


if __name__ == "__main__":
    main()