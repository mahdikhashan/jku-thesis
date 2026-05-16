import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill_lizard_llama_3_2_1B import swap_attention, STAGE2_CKPT

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

prompts = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nName three planets in our solar system.\n\n### Response:\n",
    "### Instruction:\nWrite a short greeting.\n\n### Response:\n",
    "### Instruction:\nWhat does the word 'ephemeral' mean?\n\n### Response:\n",
]

GEN_KWARGS = dict(
    max_new_tokens=50,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

lizard = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", dtype=torch.bfloat16
).to("cuda")
lizard = swap_attention(lizard)
lizard.load_state_dict(torch.load(STAGE2_CKPT, map_location="cuda"), strict=False)
lizard.eval()

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = lizard.generate(**inputs, **GEN_KWARGS, use_cache=False)
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Q: {p.split('### Instruction:')[1].split('###')[0].strip()}")
    print(f"A: {response.strip()}")
    print("-" * 40)
