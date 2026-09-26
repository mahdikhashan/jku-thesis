import sys

import torch
from transformers import AutoTokenizer

import config as C
from evaluate import load

PROMPTS = [
    "What is the capital of France?",
    "Name three planets in our solar system.",
    "Write a short greeting.",
    "What does the word 'ephemeral' mean?",
]


def main(models):
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    for which in models:
        model = load(which)
        print("=" * 60, which.upper(), sep="\n")
        for question in PROMPTS:
            prompt = f"### Instruction:\n{question}\n\n### Response:\n"
            ids = tokenizer(prompt, return_tensors="pt").to(C.DEVICE)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id,
                                     use_cache=which != "lizard")
            answer = tokenizer.decode(out[0, ids["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
            print(f"Q: {question}\nA: {answer.strip()}\n" + "-" * 40)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["llama", "lizard"])
