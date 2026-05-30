import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# Import from your training script
from distill_lizard_llama_3_2_1B import LizardAttention, swap_attention, STAGE2_CKPT

# Build the Lizard model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    dtype=torch.bfloat16,
).to("cuda")
model = swap_attention(model)
state = torch.load(STAGE2_CKPT, map_location="cuda")
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"missing keys: {len(missing)}, unexpected: {len(unexpected)}")
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Wrap in HFLM for lm-eval
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)

# Evaluate
results = simple_evaluate(
    model=lm,
    tasks=["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"],
    num_fewshot=0,
)

# Print results
for task, metrics in results["results"].items():
    print(f"{task}: {metrics}")
