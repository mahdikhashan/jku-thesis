import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


class LizardAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size=128, num_meta_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.num_meta_tokens = num_meta_tokens
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_gamma = nn.Linear(hidden_size, 1, bias=True)
        self.meta_tokens = nn.Parameter(torch.randn(num_meta_tokens))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self, 
        hidden_states, 
        attention_mask=None, 
        position_ids=None, 
        past_key_value=None, 
        output_attentions=False, 
        use_cache=False,
        **kwargs
    ):
        # Hugging Face passes input tensors as 'hidden_states'
        batch, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # --- Gated Linear Attention Path ---
        gate_logits = self.W_gamma(hidden_states).squeeze(-1)
        gates = torch.sigmoid(gate_logits)
        
        q_phi, k_phi = F.silu(q), F.silu(k)
        y_gate = torch.zeros_like(q)
        S = torch.zeros((batch, self.num_heads, self.head_dim, self.head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        
        for i in range(seq_len):
            g_i = gates[:, i].view(batch, 1, 1, 1)
            k_i, v_i = k_phi[:, i].unsqueeze(-1), v[:, i].unsqueeze(-2)
            S = g_i * S + torch.matmul(k_i, v_i)
            y_gate[:, i] = torch.matmul(q_phi[:, i].unsqueeze(-2), S).squeeze(-2)
            
        # --- Anchor Window Attention Path ---
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (self.head_dim ** 0.5)
        
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        mask = torch.tril(mask)
        mask = torch.triu(mask, diagonal=-self.window_size + 1)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        exp_scores = torch.exp(attn_scores).masked_fill(~mask, 0.0)
        meta_sink_mass = torch.sum(torch.exp(self.meta_tokens))
        denominator = exp_scores.sum(dim=-1, keepdim=True) + meta_sink_mass
        
        y_anchor = torch.einsum('bhqk,bkhd->bqhd', exp_scores / denominator, v)
        
        # Combine
        y_lizard = (y_gate + self.alpha * y_anchor).contiguous().view(batch, seq_len, self.hidden_size)

        # Return signature expected by this version of LlamaDecoderLayer: (output, weights)
        return self.o_proj(y_lizard), None


def replace_with_lizard(model):
    """Replaces all Llama attention layers with Lizard Attention."""
    config = model.config
    # Detect the model's loaded precision (bfloat16 in this case)
    target_dtype = next(model.parameters()).dtype

    for layer in model.model.layers:
        new_attn = LizardAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            window_size=128,
            num_meta_tokens=4
        )
        # Cast the newly initialized fp32 weights to the target dtype
        layer.self_attn = new_attn.to(dtype=target_dtype)

    return model

# ==========================================
# 2. Alpaca Dataset Packing Logic
# ==========================================
def format_alpaca_prompt(example):
    """Formats Alpaca JSON into a standard instruction prompt."""
    if example.get("input", "") != "":
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": prompt}

def prepare_alpaca_packed_dataloader(tokenizer, seq_len, batch_size):
    """Loads, formats, tokenizes, and packs the Alpaca Cleaned dataset."""
    print("Loading yahma/alpaca-cleaned dataset...")
    raw_dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    # 1. Format the prompts
    print("Formatting prompts...")
    formatted_dataset = raw_dataset.map(format_alpaca_prompt, num_proc=4, remove_columns=raw_dataset.column_names)

    # 2. Tokenize and append EOS
    print("Tokenizing...")
    def tokenize_function(examples):
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(texts, add_special_tokens=False)

    tokenized_datasets = formatted_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # 3. Pack the sequences into uniform blocks of `seq_len`
    print(f"Packing sequences into {seq_len}-token chunks...")
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // seq_len) * seq_len
        
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    packed_dataset = tokenized_datasets.map(group_texts, batched=True, num_proc=4, batch_size=1000)
    packed_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    
    print(f"Total packed sequences: {len(packed_dataset)}")
    return DataLoader(packed_dataset, batch_size=batch_size, shuffle=True)

# ==========================================
# 3. Training Stages
# ==========================================
def train_stage_1(teacher, student, dataloader, optimizer, device):
    """Stage 1: Attention Distillation (MSE Loss on Hidden States)"""
    teacher.eval()
    student.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Stage 1: Distillation")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        
        with torch.no_grad():
            teacher_outputs = teacher(input_ids, output_hidden_states=True, return_dict=True)
            
        student_outputs = student(input_ids, output_hidden_states=True, return_dict=True)
        
        loss = 0.0
        for l in range(1, len(teacher_outputs.hidden_states)):
            y_teacher = teacher_outputs.hidden_states[l]
            y_student = student_outputs.hidden_states[l]
            loss += F.mse_loss(y_student, y_teacher)
            
        loss = loss / (len(teacher_outputs.hidden_states) - 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'mse_loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def train_stage_2(peft_model, dataloader, optimizer, scheduler, device):
    """Stage 2: Causal Language Modeling via LoRA"""
    peft_model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Stage 2: LoRA Alignment")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = peft_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'lm_loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "meta-llama/Llama-3.2-1B"
    stage_1_save_path = "./lizard-llama-3.2-1b-stage1"
    stage_2_save_path = "./lizard-llama-3.2-1b-alpaca-final"
    seq_len = 2048
    batch_size = 4

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    student = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    print("Injecting Lizard Attention...")
    student = replace_with_lizard(student).to(device)

    dataloader = prepare_alpaca_packed_dataloader(tokenizer, seq_len, batch_size)

    # ==========================================
    # STAGE 1: Attention Approximation
    # ==========================================
    print("\n--- Starting Stage 1: Attention Distillation ---")
    optimizer_s1 = optim.AdamW(student.parameters(), lr=3e-4)

    # Train Stage 1
    train_stage_1(teacher, student, dataloader, optimizer_s1, device)

    # Save Stage 1 Base Model
    print(f"\nSaving Stage 1 distilled model to {stage_1_save_path}...")
    # Because we monkey-patched the model, we save the state_dict directly
    # to avoid HuggingFace config conflicts on reload.
    torch.save(student.state_dict(), f"{stage_1_save_path}_weights.pth")
    tokenizer.save_pretrained(stage_1_save_path)

    # Free up VRAM (Teacher is no longer needed for Stage 2)
    print("Unloading Teacher model to free up VRAM...")
    del teacher
    torch.cuda.empty_cache()
    gc.collect()

    # ==========================================
    # STAGE 2: LoRA Alignment
    # ==========================================
    print("\n--- Starting Stage 2: LoRA Fine-Tuning ---")

    # (Optional) If running Stage 2 in a completely separate script later,
    # you would load it like this:
    # student = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # student = replace_with_lizard(student)
    # student.load_state_dict(torch.load(f"{stage_1_save_path}_weights.pth"))
    # student = student.to(device)

    # Configure LoRA for Stage 2
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "W_gamma"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Wrap the Stage 1 distilled student with LoRA
    peft_student = get_peft_model(student, lora_config).to(device)
    peft_student.print_trainable_parameters()

    optimizer_s2 = optim.AdamW(peft_student.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer_s2, num_warmup_steps=100, num_training_steps=len(dataloader)
    )

    # Train Stage 2
    train_stage_2(peft_student, dataloader, optimizer_s2, scheduler, device)

    print(f"\nTraining complete. Saving final LoRA adapters to {stage_2_save_path}...")
    peft_student.save_pretrained(stage_2_save_path)
    tokenizer.save_pretrained(stage_2_save_path)


if __name__ == "__main__":
    main()