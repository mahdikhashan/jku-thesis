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

import math
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


# ============================================================
# HYPERPARAMETERS (edit here)
# ============================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SUBSET = 50_000
SEQ_LEN = 2048

# Lizard architecture
FEATURE_DIM = 128
WINDOW_SIZE = 128
NUM_META_TOKENS = 4

# Training (shared)
MICRO_BATCH = 1
GLOBAL_BATCH = 8
GRAD_ACCUM = GLOBAL_BATCH // MICRO_BATCH
NUM_EPOCHS = 2
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.99)
ADAM_EPS = 1e-8
WARMUP_RATIO = 0.1
LOG_EVERY = 1 #25

# Stage-specific
STAGE1_LR = 1e-3
STAGE2_LR = 5e-4

# LoRA (Stage 2 only)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj"]

# I/O
CKPT_DIR = Path("./checkpoints")
STAGE1_CKPT = CKPT_DIR / "stage1_lizard.pt"
STAGE2_CKPT = CKPT_DIR / "stage2_lizard_full.pt"
TOKENIZER_DIR = CKPT_DIR / "tokenizer"

# System
DEVICE = "cuda"
DTYPE = torch.bfloat16
SEED = 42

# WandB
WANDB_PROJECT = "lizard-1b"


# Names of parameters that belong to the Lizard module (used for freeze logic)
LIZARD_PARAM_KEYS = ("phi_q", "phi_k", "W_gamma", "meta_tokens", "alpha_blend")


# ============================================================
# LIZARD ATTENTION MODULE
# ============================================================
class LizardAttention(nn.Module):
    """Drop-in replacement for LlamaAttention.

    Output = GLA(x) + alpha * AnchorWindow(x)
      - GLA: globally-aware gated linear attention (RoPE-free, recurrent)
      - AnchorWindow: local softmax attention with meta-memory denominator tokens
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads          # 32 for 1B
        self.num_kv_heads = config.num_key_value_heads       # 8 for 1B (GQA)
        self.head_dim = self.hidden_size // self.num_heads   # 64
        self.kv_groups = self.num_heads // self.num_kv_heads # 4

        # Standard projections (copied from teacher at swap time)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Lizard-added: Hedgehog feature maps phi_q, phi_k : head_dim -> 2 * FEATURE_DIM
        self.phi_q = nn.Linear(self.head_dim, FEATURE_DIM, bias=False)
        self.phi_k = nn.Linear(self.head_dim, FEATURE_DIM, bias=False)

        # Scalar gate (shared across heads, per token): hidden -> 1
        self.W_gamma = nn.Linear(self.hidden_size, 1, bias=False)

        # Meta-memory token logits (denominator-only sinks for the window branch)
        self.meta_tokens = nn.Parameter(torch.zeros(NUM_META_TOKENS))

        # Learnable blend coefficient
        self.alpha_blend = nn.Parameter(torch.tensor(1.0))

        # Sensible init: feature maps small, gate centered at sigmoid(0) = 0.5
        nn.init.normal_(self.phi_q.weight, std=0.02)
        nn.init.normal_(self.phi_k.weight, std=0.02)
        nn.init.zeros_(self.W_gamma.weight)

    def hedgehog(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # x: (B, H, L, head_dim) -> (B, H, L, 2 * FEATURE_DIM)
        xw = proj(x)
        return torch.cat([torch.exp(xw), torch.exp(-xw)], dim=-1)

    def gla_branch(
        self,
        phi_q: torch.Tensor,
        phi_k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """Stable parallel-form gated linear attention.

        phi_q, phi_k : (B, H, L, F)
        v            : (B, H, L, D)
        gate         : (B, L) in (0, 1]   (sigmoid output, shared across heads)
        returns      : (B, H, L, D)

        Stability: instead of scaling Q by C and K by 1/C (which under/overflows),
        we compute pairwise scaling exp(cum_log_gate[t] - cum_log_gate[s]) which
        is always in (0, 1] for causal s <= t.
        """
        B, H, L, _ = phi_q.shape
        with torch.amp.autocast(device_type="cuda", enabled=False):
            phi_q_f = phi_q.float()
            phi_k_f = phi_k.float()
            v_f = v.float()
            gate_f = gate.float()

            log_gate = torch.log(gate_f.clamp(min=1e-6))            # (B, L), <= 0
            cum_log_gate = torch.cumsum(log_gate, dim=-1)            # (B, L)
            # G[b, t, s] = exp(cum[t] - cum[s]) for causal s <= t
            # G = torch.exp(cum_log_gate.unsqueeze(-1) - cum_log_gate.unsqueeze(-2))  # (B, L, L)
            # causal = torch.tril(torch.ones(L, L, device=phi_q.device, dtype=torch.bool))
            # G = G * causal.unsqueeze(0)                              # zero strict upper-tri
            diff = cum_log_gate.unsqueeze(-1) - cum_log_gate.unsqueeze(-2)  # (B, L, L)
            causal = torch.tril(torch.ones(L, L, device=phi_q.device, dtype=torch.bool))
            diff = diff.masked_fill(~causal.unsqueeze(0), float('-inf'))
            G = torch.exp(diff)  # safe: ≤ 1 on causal entries, 0 (from exp(-inf)) on masked entries

            scores = torch.matmul(phi_q_f, phi_k_f.transpose(-2, -1))  # (B, H, L, L)
            scores = scores * G.unsqueeze(1)                          # broadcast over heads
            num = torch.matmul(scores, v_f)                           # (B, H, L, D)
            denom = scores.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, H, L, 1)
            out = num / denom
        return out.to(v.dtype)

    def awa_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sliding-window softmax attention with meta tokens in denominator only.

        q, k, v: (B, H, L, D)
        """
        B, H, L, D = q.shape
        scale = 1.0 / math.sqrt(D)

        idx = torch.arange(L, device=q.device)
        # Position i attends to j in [i - w + 1, i]
        valid = (idx.unsqueeze(0) <= idx.unsqueeze(1)) & (
            (idx.unsqueeze(1) - idx.unsqueeze(0)) < WINDOW_SIZE
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale         # (B, H, L, L)
        scores = scores.masked_fill(~valid, float("-inf"))

        # Numerically stable softmax with extra denominator terms
        max_scores = scores.max(dim=-1, keepdim=True).values
        # Replace -inf rows (no valid tokens shouldn't happen but be safe)
        max_scores = torch.where(torch.isinf(max_scores), torch.zeros_like(max_scores), max_scores)
        exp_scores = torch.exp(scores - max_scores).masked_fill(~valid, 0.0)
        num = torch.matmul(exp_scores, v)

        denom_local = exp_scores.sum(dim=-1, keepdim=True)            # (B, H, L, 1)
        # meta_tokens are logits in same space as scores; rescale by max_scores
        meta_logits = self.meta_tokens.float().view(1, 1, 1, -1)      # (1, 1, 1, m)
        denom_meta = torch.exp(meta_logits - max_scores).sum(dim=-1, keepdim=True)  # (B, H, L, 1)

        denom = (denom_local + denom_meta).clamp(min=1e-6)
        return num / denom

    def forward(self, hidden_states, **kwargs):
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expand (no RoPE -- Lizard removes it)
        k = k.repeat_interleave(self.kv_groups, dim=1)
        v = v.repeat_interleave(self.kv_groups, dim=1)

        # Branches
        phi_q = self.hedgehog(q, self.phi_q)
        phi_k = self.hedgehog(k, self.phi_k)
        gate = torch.sigmoid(self.W_gamma(hidden_states)).squeeze(-1)  # (B, L)
        y_gla = self.gla_branch(phi_q, phi_k, v, gate)
        if torch.isnan(y_gla).any() or torch.isinf(y_gla).any():
            print(f"[layer {self.layer_idx}] GLA produced NaN/Inf. "
                  f"phi_q max: {phi_q.abs().max().item():.2e}, "
                  f"phi_k max: {phi_k.abs().max().item():.2e}, "
                  f"gate min: {gate.min().item():.4f}")

        y_awa = self.awa_branch(q, k, v)
        if torch.isnan(y_awa).any() or torch.isinf(y_awa).any():
            print(f"[layer {self.layer_idx}] AWA produced NaN/Inf")

        out = y_gla + self.alpha_blend * y_awa
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.o_proj(out.to(hidden_states.dtype))

        # Return tuple matching LlamaAttention signature
        return out, None, None


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
                y_pred, _, _ = layer.self_attn(x)
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

    # Required for grad checkpointing through peft
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

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
    model = model.merge_and_unload()
    full_sd = {n: p.detach().cpu() for n, p in model.named_parameters()}
    torch.save(full_sd, STAGE2_CKPT)
    print(f"  Stage 2 full state dict -> {STAGE2_CKPT}")
    print(f"  To reload: load base Llama-3.2-1B, call swap_attention(m), then m.load_state_dict(torch.load(STAGE2_CKPT), strict=False)")
    wandb.finish()


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(SEED)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_distill()
    stage2_finetune()
    print("Done.")


if __name__ == "__main__":
    main()