from config import *

import torch
state = torch.load(STAGE2_CKPT, map_location="cpu")

print(f"Total keys in checkpoint: {len(state)}")
print(f"Keys containing 'meta': {[k for k in state.keys() if 'meta' in k][:5]}")
print(f"Keys containing 'alpha': {[k for k in state.keys() if 'alpha' in k][:5]}")
print(f"Keys containing 'phi_q': {[k for k in state.keys() if 'phi_q' in k][:5]}")
print(f"Keys containing 'W_gamma': {[k for k in state.keys() if 'W_gamma' in k][:5]}")

# Check values directly
if any('meta_tokens' in k for k in state.keys()):
    sample_key = next(k for k in state.keys() if 'meta_tokens' in k)
    print(f"\nSample meta_tokens value from ckpt: {state[sample_key]}")
