import math

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from lizard_attention import from_llama


def is_lizard(name):
    return any(key in name.split(".") for key in C.LIZARD_PARAMS)


def load_llama():
    model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, dtype=C.DTYPE)
    return model.to(C.DEVICE)


def to_lizard(model):
    for block in model.model.layers:
        block.self_attn = from_llama(block.self_attn, model.config, **C.LIZARD)
    return model


def format_example(example):
    extra = ""
    if example["input"]:
        extra = f"### Input:\n{example['input']}\n\n"
    return (f"### Instruction:\n{example['instruction']}\n\n{extra}"
            f"### Response:\n{example['output']}")


def pack(token_lists, length):
    ids = [token for tokens in token_lists for token in tokens]
    n = len(ids) // length
    return torch.tensor(ids[:n * length]).view(n, length)


def make_loader(tokenizer):
    data = load_dataset(C.DATASET_NAME, split="train")
    data = data.shuffle(seed=C.SEED).select(range(C.NUM_EXAMPLES))
    tokens = tokenizer([format_example(e) for e in data])["input_ids"]
    chunks = pack([t + [tokenizer.eos_token_id] for t in tokens], C.SEQ_LEN)
    return DataLoader(chunks, batch_size=C.MICRO_BATCH, shuffle=True,
                      drop_last=True,
                      generator=torch.Generator().manual_seed(C.SEED))


def lr_factor(step, total):
    warmup = max(1, int(C.WARMUP_RATIO * total))
    if step < warmup:
        return step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return C.MIN_LR_RATIO + (1 - C.MIN_LR_RATIO) * cosine


def log(stage, loss, named):
    values = {"loss": loss}
    for key in ("alpha_blend", "meta_tokens"):
        means = [p.detach().mean() for n, p in named if n.endswith(key)]
        values[key] = torch.stack(means).mean().item()
    print(stage, values)
    if wandb.run is not None:
        wandb.log({f"{stage}/{k}": v for k, v in values.items()})


def train(named, step_fn, loader, lr, stage):
    params = [p for _, p in named]
    total = C.EPOCHS * (len(loader) // C.GRAD_ACCUM)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=C.BETAS, eps=C.EPS,
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, total))
    for _ in range(C.EPOCHS):
        optimizer.zero_grad()
        loss = 0.0
        for i, batch in enumerate(loader):
            loss += step_fn(batch.to(C.DEVICE))
            if (i + 1) % C.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, C.GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                log(stage, loss, named)
                loss = 0.0


def stage1(teacher, loader):
    teacher.eval().requires_grad_(False)
    students = [from_llama(block.self_attn, teacher.config, **C.LIZARD)
                for block in teacher.model.layers]
    for student in students:
        for name, p in student.named_parameters():
            p.requires_grad = is_lizard(name)
    captured = {}
    for i, block in enumerate(teacher.model.layers):
        def hook(module, args, kwargs, output, i=i):
            x = args[0] if args else kwargs["hidden_states"]
            captured[i] = (x, output[0])
        block.self_attn.register_forward_hook(hook, with_kwargs=True)

    def step_fn(ids):
        with torch.no_grad():
            teacher.model(input_ids=ids, use_cache=False)
        total = 0.0
        for i, student in enumerate(students):
            x, target = captured[i]
            loss = (student(x)[0] - target).pow(2).sum()
            loss = loss / len(students) / C.GRAD_ACCUM
            loss.backward()
            total += loss.item()
        return total

    named = [(n, p) for s in students for n, p in s.named_parameters()
             if p.requires_grad]
    train(named, step_fn, loader, C.STAGE1_LR, "stage1")
    return {f"model.layers.{i}.self_attn.{n}": p.detach().cpu()
            for i, s in enumerate(students)
            for n, p in s.named_parameters() if is_lizard(n)}


def stage2(model, lizard_state, loader):
    model = to_lizard(model)
    _, unexpected = model.load_state_dict(lizard_state, strict=False)
    assert not unexpected, unexpected
    if C.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=C.LORA_RANK, lora_alpha=C.LORA_ALPHA,
                      lora_dropout=0.0, target_modules=C.LORA_TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora).train()
    for name, p in model.named_parameters():
        if is_lizard(name):
            p.requires_grad = True

    def step_fn(ids):
        out = model(input_ids=ids, labels=ids, use_cache=False)
        loss = out.loss / C.GRAD_ACCUM
        loss.backward()
        return loss.item()

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    train(named, step_fn, loader, C.STAGE2_LR, "stage2")
    return model.merge_and_unload()


def main():
    torch.manual_seed(C.SEED)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    wandb.init(project=C.WANDB_PROJECT)
    loader = make_loader(AutoTokenizer.from_pretrained(C.MODEL_NAME))
    if C.STAGE1_CKPT.exists():
        lizard_state = torch.load(C.STAGE1_CKPT)
    else:
        lizard_state = stage1(load_llama(), loader)
        torch.save(lizard_state, C.STAGE1_CKPT)
    model = stage2(load_llama(), lizard_state, loader)
    torch.save(model.state_dict(), C.STAGE2_CKPT)
    wandb.finish()


if __name__ == "__main__":
    main()
