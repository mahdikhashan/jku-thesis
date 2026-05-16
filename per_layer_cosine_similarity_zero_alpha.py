"""Per-layer cosine similarity test with alpha ablation.

Runs the cosine analysis twice on the same model:
  Test A: alpha=0 (GLA only)
  Test B: alpha=original trained values (both branches)

Compares which configuration better approximates the teacher's softmax attention.

Diagnostic:
  - If Test A (GLA only) cosine >> Test B: the summing of branches is the bug,
    each branch is over-producing magnitude
  - If Test B cosine >> Test A: branches complement each other (architecture works)
  - If both are similar and low: deeper structural issue with the attention math
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill_lizard_llama_3_2_1B import swap_attention, STAGE2_CKPT

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DTYPE = torch.bfloat16
DEVICE = "cuda"

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree when the fox came running through the meadow.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
    "The capital of France is Paris, a city renowned for its art, culture, and history. The Eiffel Tower stands as one of its most iconic landmarks.",
]


def make_hook(store, idx):
    def hook(_, __, out):
        tensor = out[0] if isinstance(out, tuple) else out
        store[idx] = tensor.detach()
    return hook


def register_hooks(model, store):
    handles = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(store, i))
        handles.append(h)
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


def compute_per_layer_stats(teacher, lizard, tokenizer, num_layers, label):
    """Run all prompts and return per-layer mean cosine and norm ratio."""
    print(f"\n{'='*70}")
    print(f"Computing: {label}")
    print(f"{'='*70}")

    per_layer_cos = [[] for _ in range(num_layers)]
    per_layer_norm_ratio = [[] for _ in range(num_layers)]

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"  Prompt {p_idx + 1}/{len(PROMPTS)}: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        teacher_outs, lizard_outs = {}, {}
        h1 = register_hooks(teacher, teacher_outs)
        h2 = register_hooks(lizard, lizard_outs)

        with torch.no_grad():
            teacher(**inputs)
            lizard(**inputs)

        remove_hooks(h1)
        remove_hooks(h2)

        for i in range(num_layers):
            t = teacher_outs[i].float().flatten()
            l = lizard_outs[i].float().flatten()
            cos = F.cosine_similarity(t, l, dim=0).item()
            ratio = l.norm().item() / t.norm().clamp(min=1e-8).item()
            per_layer_cos[i].append(cos)
            per_layer_norm_ratio[i].append(ratio)

    # Compute means
    mean_cos = [sum(c) / len(c) for c in per_layer_cos]
    mean_ratio = [sum(r) / len(r) for r in per_layer_norm_ratio]
    return mean_cos, mean_ratio


def print_comparison(num_layers, cos_a, ratio_a, cos_b, ratio_b):
    """Print side-by-side table comparing alpha=0 vs alpha=original."""
    print("\n" + "=" * 90)
    print(f"{'Layer':>5}  "
          f"{'A: GLA only':>18}  {'B: both branches':>22}  "
          f"{'A wins by':>12}")
    print(f"{'':>5}  {'cos / norm_ratio':>18}  {'cos / norm_ratio':>22}  "
          f"{'':>12}")
    print("-" * 90)
    for i in range(num_layers):
        delta = cos_a[i] - cos_b[i]
        winner = "A" if delta > 0.01 else ("B" if delta < -0.01 else "≈")
        print(f"{i:>5}  "
              f"{cos_a[i]:>7.4f} / {ratio_a[i]:>6.3f}  "
              f"{cos_b[i]:>9.4f} / {ratio_b[i]:>6.3f}  "
              f"{delta:>+8.4f} ({winner})")
    print("-" * 90)

    overall_a = sum(cos_a) / len(cos_a)
    overall_b = sum(cos_b) / len(cos_b)
    print(f"Overall mean cosine:   A (GLA only) = {overall_a:.4f}    "
          f"B (both branches) = {overall_b:.4f}")
    print(f"Overall mean norm:     A (GLA only) = {sum(ratio_a)/len(ratio_a):.4f}    "
          f"B (both branches) = {sum(ratio_b)/len(ratio_b):.4f}")
    print(f"Delta (A - B): {overall_a - overall_b:+.4f}")

    print("\n" + "=" * 90)
    print("Interpretation:")
    if overall_a - overall_b > 0.05:
        print("  → GLA ALONE is closer to teacher than the combination.")
        print("  → BUG: summing the two branches over-produces magnitude.")
        print("  → Fix: retrain with smaller alpha init, or larger LoRA rank to absorb.")
    elif overall_b - overall_a > 0.05:
        print("  → BOTH BRANCHES is closer to teacher than GLA alone.")
        print("  → Branches genuinely complement each other; architecture works.")
        print("  → The cosine ~0.6 gap to teacher must come from another source.")
    else:
        print("  → Neither config matches the teacher well (both around 0.55-0.60).")
        print("  → The architecture itself may not approximate softmax at this scale.")
        print("  → Fix candidates: Hedgehog feature_dim, different feature map, more training.")
    print("=" * 90)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load teacher
    print("Loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load Lizard
    print("Loading Lizard...")
    lizard = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE).to(DEVICE)
    lizard = swap_attention(lizard)
    state = torch.load(STAGE2_CKPT, map_location=DEVICE)
    missing, unexpected = lizard.load_state_dict(state, strict=False)
    print(f"  loaded checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected")
    lizard.eval()
    for p in lizard.parameters():
        p.requires_grad = False

    num_layers = len(teacher.model.layers)

    # Save original alpha values so we can restore them
    original_alphas = [layer.self_attn.alpha_blend.item() for layer in lizard.model.layers]
    print(f"\nOriginal trained alpha_blend values: "
          f"min={min(original_alphas):.3f}, max={max(original_alphas):.3f}, "
          f"mean={sum(original_alphas)/len(original_alphas):.3f}")

    # ---- Test A: alpha = 0 (GLA only) ----
    for layer in lizard.model.layers:
        layer.self_attn.alpha_blend.data.fill_(0.0)
    cos_a, ratio_a = compute_per_layer_stats(
        teacher, lizard, tokenizer, num_layers,
        "Test A: alpha=0 (GLA branch only)"
    )

    # ---- Test B: alpha = original trained values (both branches) ----
    for layer, orig in zip(lizard.model.layers, original_alphas):
        layer.self_attn.alpha_blend.data.fill_(orig)
    cos_b, ratio_b = compute_per_layer_stats(
        teacher, lizard, tokenizer, num_layers,
        "Test B: alpha=original (GLA + alpha*AWA)"
    )

    # ---- Side-by-side comparison ----
    print_comparison(num_layers, cos_a, ratio_a, cos_b, ratio_b)


if __name__ == "__main__":
    main()
