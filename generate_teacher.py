import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill_lizard_llama_3_2_1B import swap_attention, STAGE2_CKPT

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Alpaca instruction format — matches training data distribution
prompt = (
    "### Instruction:\n"
    "What is the capital of France?\n\n"
    "### Response:\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

GEN_KWARGS = dict(
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

# 1) Teacher
teacher = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16
).to("cuda").eval()

with torch.no_grad():
    out_t = teacher.generate(**inputs, **GEN_KWARGS)
print("=" * 60)
print("TEACHER:")
print(tokenizer.decode(out_t[0], skip_special_tokens=True))
print()

del teacher
torch.cuda.empty_cache()

# 2) Lizard
lizard = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16
).to("cuda")
lizard = swap_attention(lizard)
lizard.load_state_dict(torch.load(STAGE2_CKPT, map_location="cuda"), strict=False)
lizard.eval()

with torch.no_grad():
    out_l = lizard.generate(
        **inputs,
        **GEN_KWARGS,
        use_cache=False,   # Lizard doesn't support KV caching
    )
print("=" * 60)
print("LIZARD:")
print(tokenizer.decode(out_l[0], skip_special_tokens=True))