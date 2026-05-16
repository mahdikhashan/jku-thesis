"""Per-layer cosine similarity between teacher and trained Lizard attention outputs.

Compares the attention layer outputs (post-o_proj, just before the residual+MLP)
between the frozen Llama-3.2-1B teacher and your trained Lizard model.

High cosine similarity (>0.95) across all layers = Stage 1 distillation worked well.
Declining similarity with depth = compounding drift, Stage 1 has structural issues.
Low similarity everywhere = distillation didn't converge.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill_lizard_llama_3_2_1B import swap_attention, STAGE2_CKPT

MODEL_NAME = "meta-llama/Llama-3.2-1B"
DTYPE = torch.bfloat16
DEVICE = "cuda"

# A few diverse prompts so we don't draw conclusions from one input
PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The dog was sleeping under a tree when the fox came running through the meadow.",
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell.",
    "The capital of France is Paris, a city renowned for its art, culture, and history. The Eiffel Tower stands as one of its most iconic landmarks.",
]


def make_hook(store, idx):
    def hook(_, __, out):
        # Llama self_attn returns (attn_output, attn_weights) — take the output tensor
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

    print("\n=== Test A: alpha=0 (GLA only) ===")
    for layer in lizard.model.layers:
        layer.self_attn.alpha_blend.data.fill_(0.0)

    
    num_layers = len(teacher.model.layers)

    # Aggregate cosine sims per layer across prompts
    per_layer_cos = [[] for _ in range(num_layers)]
    per_layer_norm_ratio = [[] for _ in range(num_layers)]  # |lizard| / |teacher|

    for p_idx, prompt in enumerate(PROMPTS):
        print(f"\nPrompt {p_idx + 1}/{len(PROMPTS)}: {prompt[:60]}...")
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

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Layer':>5}  {'cosine sim':>12}  {'|liz| / |tea|':>15}  {'verdict':>20}")
    print("-" * 70)
    for i in range(num_layers):
        mean_cos = sum(per_layer_cos[i]) / len(per_layer_cos[i])
        mean_ratio = sum(per_layer_norm_ratio[i]) / len(per_layer_norm_ratio[i])

        if mean_cos > 0.95:
            verdict = "match"
        elif mean_cos > 0.80:
            verdict = "partial"
        elif mean_cos > 0.50:
            verdict = "weak"
        else:
            verdict = "broken"

        print(f"{i:>5}  {mean_cos:>12.4f}  {mean_ratio:>15.4f}  {verdict:>20}")

    # Overall stats
    all_cos = [c for layer in per_layer_cos for c in layer]
    overall_mean = sum(all_cos) / len(all_cos)
    early = sum(per_layer_cos[i][0] for i in range(min(4, num_layers))) / min(4, num_layers)
    late = sum(per_layer_cos[i][0] for i in range(max(0, num_layers - 4), num_layers)) / min(4, num_layers)

    print("-" * 70)
    print(f"Overall mean cosine: {overall_mean:.4f}")
    print(f"Early layers (0-3) mean: {early:.4f}")
    print(f"Late layers (-4:-1) mean: {late:.4f}")
    print(f"Drift (early - late): {early - late:+.4f}")


if __name__ == "__main__":
    main()
