from pathlib import Path

import torch

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "yahma/alpaca-cleaned"
NUM_EXAMPLES = 50_000
SEQ_LEN = 2048
DTYPE = torch.float32
DEVICE = "cuda"
SEED = 42

LIZARD = {"window": 128, "num_meta": 4, "feature_dim": 128}
LIZARD_PARAMS = ("phi_q", "phi_k", "W_gamma", "meta_tokens", "alpha_blend")

MICRO_BATCH = 1
GRAD_ACCUM = 8
EPOCHS = 2
WARMUP_RATIO = 0.1
MIN_LR_RATIO = 0.1
GRAD_CLIP = 1.0
BETAS = (0.9, 0.99)
EPS = 1e-8
WEIGHT_DECAY = 0.0
STAGE1_LR = 1e-3
STAGE2_LR = 5e-4

LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj"]
GRADIENT_CHECKPOINTING = True

OUT_DIR = Path("checkpoints")
STAGE1_CKPT = OUT_DIR / "stage1_lizard.pt"
STAGE2_CKPT = OUT_DIR / "stage2_model.pt"
WANDB_PROJECT = "lizard-1b"
