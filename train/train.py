"""
Generic Unsloth fine-tuning driver.

Usage:
    python train/train.py train/configs/alpaca-smoke.yaml
    python train/train.py train/configs/my-config.yaml --max-steps 1000 --name v2

The config is YAML; CLI flags override individual fields. See
train/configs/README.md for the full schema and example configs.

Output goes to <output_dir>/<name>/ on the network volume:
  - adapter/      LoRA weights you'd load for inference
  - checkpoint-N/ intermediate checkpoints (kept per save_total_limit)

After training, point chat.py or infer.py at the adapter directory.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig


# --- defaults ---------------------------------------------------------------
DEFAULTS = {
    "max_seq_length": 2048,
    "lora": {
        "r": 16,
        "alpha": 16,
        "dropout": 0.0,
        "use_gradient_checkpointing": "unsloth",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    },
    "max_steps": 60,
    "num_train_epochs": None,    # alternative to max_steps
    "batch_size": 2,             # per_device_train_batch_size
    "grad_accum": 4,
    "learning_rate": 2.0e-4,
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "lr_scheduler": "linear",
    "logging_steps": 1,
    "save_steps": 60,
    "save_total_limit": 2,
    "seed": 3407,
    "output_dir": "/workspace/runs",
    "wandb_project": "runpod-unsloth",
    "data": {
        "format": "alpaca",
        "split": "train",
        "text_field": "text",  # used when format=raw
        # Optional eval inputs (any one enables held-out evaluation):
        #   eval_split:    HF split string, e.g. "test" or "train[2000:2200]"
        #   eval_path:     jsonl path for a separate eval file
        #   eval_fraction: float in (0, 1) — random holdout from train
    },
    "eval": {
        "steps": None,       # how often to evaluate; falls back to save_steps
        "batch_size": None,  # per_device_eval_batch_size; falls back to batch_size
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# --- prompt formats ---------------------------------------------------------
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)


def format_alpaca(ds, tokenizer):
    """Expects fields: instruction, input, output."""
    eos = tokenizer.eos_token

    def fmt(batch):
        return {"text": [
            ALPACA_TEMPLATE.format(instruction=i, input=x or "", output=o) + eos
            for i, x, o in zip(batch["instruction"], batch["input"], batch["output"])
        ]}

    return ds.map(fmt, batched=True, remove_columns=ds.column_names)


def format_chat(ds, tokenizer):
    """Expects field 'messages' or 'conversations' as a list of {role, content}."""
    field = "messages" if "messages" in ds.column_names else "conversations"
    if field not in ds.column_names:
        sys.exit(f"chat format requires 'messages' or 'conversations' column; "
                 f"got: {ds.column_names}")

    def fmt(batch):
        return {"text": [
            tokenizer.apply_chat_template(msgs, tokenize=False)
            for msgs in batch[field]
        ]}

    return ds.map(fmt, batched=True, remove_columns=ds.column_names)


def format_raw(ds, tokenizer, text_field="text"):
    """Already-formatted text — just rename column to 'text' if needed."""
    if text_field == "text":
        return ds
    return ds.rename_column(text_field, "text")


def _apply_format(ds, cfg, tokenizer):
    fmt = cfg["data"].get("format", "alpaca")
    if fmt == "alpaca":
        return format_alpaca(ds, tokenizer)
    if fmt == "chat":
        return format_chat(ds, tokenizer)
    if fmt == "raw":
        return format_raw(ds, tokenizer, cfg["data"].get("text_field", "text"))
    sys.exit(f"unknown data format: {fmt!r} (expected 'alpaca', 'chat', or 'raw')")


def load_data(cfg, tokenizer):
    """Returns (train_ds, eval_ds_or_None)."""
    src = cfg["data"]["source"]
    split = cfg["data"].get("split", "train")
    eval_split = cfg["data"].get("eval_split")
    eval_path = cfg["data"].get("eval_path")
    eval_fraction = cfg["data"].get("eval_fraction")

    if src == "hf":
        train_raw = load_dataset(cfg["data"]["name"], split=split)
        eval_raw = load_dataset(cfg["data"]["name"], split=eval_split) if eval_split else None
    elif src == "jsonl":
        train_raw = load_dataset("json", data_files=cfg["data"]["path"], split="train")
        if split != "train":
            # Allow slicing for jsonl too: e.g. "train[:1000]" -> ds.select(range(1000))
            print(f"[data] jsonl source ignores split='{split}'; loaded all rows", file=sys.stderr)
        eval_raw = load_dataset("json", data_files=eval_path, split="train") if eval_path else None
    else:
        sys.exit(f"unknown data source: {src!r} (expected 'hf' or 'jsonl')")

    if eval_raw is None and eval_fraction:
        if not 0 < float(eval_fraction) < 1:
            sys.exit(f"data.eval_fraction must be in (0, 1); got {eval_fraction!r}")
        parts = train_raw.train_test_split(test_size=float(eval_fraction), seed=cfg["seed"])
        train_raw, eval_raw = parts["train"], parts["test"]

    train_ds = _apply_format(train_raw, cfg, tokenizer)
    eval_ds = _apply_format(eval_raw, cfg, tokenizer) if eval_raw is not None else None
    return train_ds, eval_ds


# --- main -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to YAML config")
    p.add_argument("--name", help="Override run name")
    p.add_argument("--max-steps", type=int, help="Override max training steps")
    p.add_argument("--epochs", type=int, help="Train for N epochs (overrides max-steps)")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--batch-size", type=int, help="Override per-device batch size")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = p.parse_args()

    with open(args.config) as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge({k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULTS.items()},
                     user_cfg)

    if args.name:        cfg["name"] = args.name
    if args.max_steps:   cfg["max_steps"] = args.max_steps; cfg["num_train_epochs"] = None
    if args.epochs:      cfg["num_train_epochs"] = args.epochs; cfg["max_steps"] = -1
    if args.lr:          cfg["learning_rate"] = args.lr
    if args.batch_size:  cfg["batch_size"] = args.batch_size

    if "name" not in cfg or "model" not in cfg or "data" not in cfg or "source" not in cfg["data"]:
        sys.exit("config must include `name`, `model`, and `data.source`")

    out_dir = f"{cfg['output_dir']}/{cfg['name']}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== run         {cfg['name']}")
    print(f"=== model       {cfg['model']}")
    print(f"=== data        {cfg['data']['source']}: "
          f"{cfg['data'].get('name') or cfg['data'].get('path')} "
          f"(format={cfg['data'].get('format', 'alpaca')})")
    print(f"=== output      {out_dir}")
    print(f"=== seq_len     {cfg['max_seq_length']}")
    if cfg.get("num_train_epochs"):
        print(f"=== epochs      {cfg['num_train_epochs']}")
    else:
        print(f"=== max_steps   {cfg['max_steps']}")
    print(f"=== batch       {cfg['batch_size']} x grad_accum {cfg['grad_accum']}")
    print(f"=== lr          {cfg['learning_rate']}")
    print()

    if not args.no_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb_project"])

    # --- Model + LoRA ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = cfg["model"],
        max_seq_length  = cfg["max_seq_length"],
        dtype           = None,
        load_in_4bit    = True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg["lora"]["r"],
        target_modules             = cfg["lora"]["target_modules"],
        lora_alpha                 = cfg["lora"]["alpha"],
        lora_dropout               = cfg["lora"]["dropout"],
        bias                       = "none",
        use_gradient_checkpointing = cfg["lora"]["use_gradient_checkpointing"],
        random_state               = cfg["seed"],
    )

    # --- Data ---
    train_ds, eval_ds = load_data(cfg, tokenizer)
    if eval_ds is not None:
        print(f"[data] loaded {len(train_ds)} train rows / {len(eval_ds)} eval rows")
    else:
        print(f"[data] loaded {len(train_ds)} rows (no eval set)")

    # --- Trainer ---
    sft_args = dict(
        output_dir                  = out_dir,
        run_name                    = cfg["name"],
        per_device_train_batch_size = cfg["batch_size"],
        gradient_accumulation_steps = cfg["grad_accum"],
        warmup_steps                = cfg["warmup_steps"],
        learning_rate               = cfg["learning_rate"],
        logging_steps               = cfg["logging_steps"],
        optim                       = "adamw_8bit",
        weight_decay                = cfg["weight_decay"],
        lr_scheduler_type           = cfg["lr_scheduler"],
        seed                        = cfg["seed"],
        bf16                        = is_bfloat16_supported(),
        fp16                        = not is_bfloat16_supported(),
        report_to                   = ("wandb" if not args.no_wandb else "none"),
        save_strategy               = "steps",
        save_steps                  = cfg["save_steps"],
        save_total_limit            = cfg["save_total_limit"],
    )
    if cfg.get("num_train_epochs"):
        sft_args["num_train_epochs"] = cfg["num_train_epochs"]
    else:
        sft_args["max_steps"] = cfg["max_steps"]

    if eval_ds is not None:
        eval_steps = cfg["eval"].get("steps") or cfg["save_steps"]
        eval_bsz = cfg["eval"].get("batch_size") or cfg["batch_size"]
        sft_args["eval_strategy"] = "steps"
        sft_args["eval_steps"] = eval_steps
        sft_args["per_device_eval_batch_size"] = eval_bsz
        print(f"=== eval        every {eval_steps} steps "
              f"(batch {eval_bsz}, ~{-(-len(eval_ds) // eval_bsz)} batches/eval)")

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = eval_ds,
        dataset_text_field = "text",
        max_seq_length     = cfg["max_seq_length"],
        packing            = False,
        args               = SFTConfig(**sft_args),
    )

    trainer.train()

    # --- Save adapter (final, separate from intermediate checkpoints) ---
    adapter_dir = f"{out_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\n[done] adapter saved to {adapter_dir}")
    print(f"       chat with it: python train/chat.py {adapter_dir}")
    print(f"       export GGUF:  python train/export_gguf.py {adapter_dir}")


if __name__ == "__main__":
    main()
