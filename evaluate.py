import sys

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

import config as C
from train import load_llama, to_lizard

ZERO_SHOT = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]


def load(which):
    model = load_llama()
    if which == "lizard":
        state = torch.load(C.STAGE2_CKPT, map_location="cpu")
        to_lizard(model).load_state_dict(state, strict=True)
    return model.eval()


def main(which):
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    lm = HFLM(pretrained=load(which), tokenizer=tokenizer, batch_size=8)
    for tasks, shots in [(ZERO_SHOT, 0), (["mmlu"], 5)]:
        results = simple_evaluate(model=lm, tasks=tasks, num_fewshot=shots)
        for task in tasks:
            print(task, results["results"][task])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lizard")
