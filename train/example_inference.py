"""
Load the LoRA adapter from a finished training run and generate.

Usage:
    python train/example_inference.py                       # default prompt
    python train/example_inference.py "Write a haiku about octopi."

Notes:
  * Loading the adapter path directly works because Unsloth's
    FastLanguageModel.from_pretrained reads the adapter's adapter_config.json
    to find the base model and pulls both in 4-bit.
  * for_inference() switches LoRA into a faster inference path (~2x).
  * Same MAX_SEQ_LEN and Alpaca template as training so the model "feels"
    like it did during fine-tuning.
"""

import sys
from unsloth import FastLanguageModel
from transformers import TextStreamer

ADAPTER = "/workspace/runs/llama3.1-8b-alpaca-demo/adapter"
MAX_SEQ_LEN = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = ADAPTER,
    max_seq_length  = MAX_SEQ_LEN,
    dtype           = None,          # auto: bf16 on Ada+ (4090)
    load_in_4bit    = True,
)
FastLanguageModel.for_inference(model)   # enables ~2x faster generation

ALPACA = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
)

instruction = sys.argv[1] if len(sys.argv) > 1 else (
    "List three high-protein vegetarian breakfast ideas."
)
context = ""

prompt = ALPACA.format(instruction, context)
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print(f"\n=== prompt ===\n{prompt}", flush=True)
print("=== response ===")
streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    **inputs,
    streamer            = streamer,
    max_new_tokens      = 256,
    do_sample           = True,
    temperature         = 0.7,
    top_p               = 0.9,
    repetition_penalty  = 1.1,
    use_cache           = True,
)
