# Training configs

Each `.yaml` here is a self-contained fine-tune recipe. Run with:

```bash
python train/train.py train/configs/<name>.yaml
```

CLI flags override individual fields without editing the file:

```bash
python train/train.py train/configs/alpaca-smoke.yaml \
    --name alpaca-smoke-v2 \
    --max-steps 1000 \
    --lr 1e-4
```

## Schema

Only `name`, `model`, `data.source`, and `data.format` are required. Everything
else has sensible defaults (see `DEFAULTS` in `train/train.py`).

```yaml
# Required
name: my-run                              # output dir = <output_dir>/<name>
model: unsloth/Meta-Llama-3.1-8B-bnb-4bit # any HF model; Unsloth-prefixed = pre-quant
data:
  source: hf                              # "hf" or "jsonl"
  name: yahma/alpaca-cleaned              # if source=hf
  # path: /workspace/datasets/my.jsonl    # if source=jsonl
  split: "train[:2000]"                   # only honored for source=hf
  format: alpaca                          # "alpaca" | "chat" | "raw"
  # text_field: text                      # if format=raw and column != "text"

# Optional — defaults shown
max_seq_length: 2048

lora:
  r: 16                                   # adapter rank
  alpha: 16
  dropout: 0.0
  use_gradient_checkpointing: unsloth     # "unsloth" | true | false
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Length: pick ONE of these
max_steps: 60                             # default
# num_train_epochs: 1                     # alternative; commenting out max_steps

# Training
batch_size: 2                             # per_device_train_batch_size
grad_accum: 4                             # gradient_accumulation_steps
learning_rate: 2.0e-4
warmup_steps: 5
weight_decay: 0.01
lr_scheduler: linear
logging_steps: 1
save_steps: 60
save_total_limit: 2
seed: 3407

# Output
output_dir: /workspace/runs               # full path = output_dir/name
wandb_project: runpod-unsloth
```

## Data formats

The `data.format` controls how each example becomes a training string:

**`alpaca`** — expects columns `instruction`, `input`, `output`. Becomes:
```
Below is an instruction... Write a response...

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}<eos>
```
Use this for instruction-tuning a base model. Default.

**`chat`** — expects column `messages` (or `conversations`) shaped like
`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
Uses the tokenizer's `apply_chat_template` to format. Use this when fine-tuning
a chat-tuned base (`*-Instruct`) on conversational data.

**`raw`** — expects a `text` column with the example already in its final
training-ready string form. Set `data.text_field` if your column is named
something else.

## Sizing notes for a 4090 (24 GB)

| model       | format | seq_len | batch x accum | LoRA r | VRAM   |
| ----------- | ------ | ------- | ------------- | ------ | ------ |
| 3B          | alpaca | 4096    | 4 x 4         | 16     | ~16 GB |
| 7B / 8B     | alpaca | 2048    | 2 x 4         | 16     | ~19 GB |
| 7B / 8B     | chat   | 4096    | 1 x 8         | 32     | ~22 GB |
| 9B (Gemma)  | alpaca | 1024    | 1 x 8         | 16     | ~22 GB |

If you OOM, raise `grad_accum` before lowering `batch_size`, and shrink
`max_seq_length` last (truncates examples).
