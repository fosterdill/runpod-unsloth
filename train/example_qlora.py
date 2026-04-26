"""
Minimal Unsloth QLoRA fine-tune sized for a 4090 (24GB).

What this shows:
  * 4-bit QLoRA on a 4-9B base model (default Llama-3.1-8B).
  * wandb logging via SFTConfig.report_to.
  * Saving outputs onto /workspace (the network volume) so they survive
    pod termination.

Run inside the pod:
    python /workspace/runpod-unsloth/train/example_qlora.py

Tuning notes for the 4090 24GB:
  * 8B QLoRA, seq_len=2048, batch=2, grad_accum=4 -> ~18-20 GB VRAM.
  * 9B (e.g. Gemma-2-9b) fits at seq_len=1024-2048 with the same settings.
  * Bump gradient_accumulation_steps before per_device_train_batch_size.
"""

import os
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

# --- model -------------------------------------------------------------------
MAX_SEQ_LEN = 2048
MODEL_NAME  = os.environ.get("BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-bnb-4bit")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,    # auto: bf16 on Ada+
    load_in_4bit    = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r               = 16,
    target_modules  = ["q_proj","k_proj","v_proj","o_proj",
                       "gate_proj","up_proj","down_proj"],
    lora_alpha      = 16,
    lora_dropout    = 0.0,
    bias            = "none",
    use_gradient_checkpointing = "unsloth",  # ~30% memory savings on 4090
    random_state    = 3407,
)

# --- data --------------------------------------------------------------------
# Swap in your own dataset path under /workspace/datasets for real runs.
ds = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")

ALPACA_FMT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
)
EOS = tokenizer.eos_token

def fmt(batch):
    return {"text": [ALPACA_FMT.format(i, x, o) + EOS
                     for i, x, o in zip(batch["instruction"], batch["input"], batch["output"])]}

ds = ds.map(fmt, batched=True)

# --- train -------------------------------------------------------------------
OUT_DIR = "/workspace/runs/llama3.1-8b-alpaca-demo"

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = ds,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LEN,
    packing         = False,
    args = SFTConfig(
        output_dir                  = OUT_DIR,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps                = 5,
        max_steps                   = 60,             # demo: ~5 min on a 4090
        learning_rate               = 2e-4,
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        bf16                        = is_bfloat16_supported(),
        fp16                        = not is_bfloat16_supported(),
        report_to                   = "wandb",        # picks up WANDB_API_KEY
        run_name                    = os.path.basename(OUT_DIR),
        save_strategy               = "steps",
        save_steps                  = 30,
        save_total_limit            = 2,
    ),
)

trainer.train()

# Save LoRA adapter on the network volume; merge to fp16 if you need it later.
model.save_pretrained(f"{OUT_DIR}/adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/adapter")
print(f"saved adapter to {OUT_DIR}/adapter")
