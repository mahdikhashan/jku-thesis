from pathlib import Path

import torch



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

USE_GRADIENT_CHECKPOINTING_STAGE1 = False
USE_GRADIENT_CHECKPOINTING_STAGE2 = True

