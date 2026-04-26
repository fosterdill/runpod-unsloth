"""
Merge a LoRA adapter into the base model and export to GGUF (for llama.cpp).

Usage:
    python train/export_gguf.py /workspace/runs/<run-name>/adapter
    python train/export_gguf.py /workspace/runs/<run-name>/adapter q5_k_m

Output:
    <run-name>/gguf/<files>.gguf

First run takes ~5-10 min (Unsloth clones+builds llama.cpp under
~/.cache/llama.cpp; both persist on the network volume).

Quant size cheat sheet for an 8B model:
    q4_k_m   ~4.6 GB   recommended for chatting on a 16 GB Mac
    q5_k_m   ~5.5 GB   slightly better quality
    q6_k     ~6.5 GB
    q8_0     ~8.5 GB   near-lossless
    f16      ~16  GB   no quantization
"""
import sys
from pathlib import Path
from unsloth import FastLanguageModel

if len(sys.argv) < 2:
    sys.exit(__doc__)

ADAPTER = sys.argv[1].rstrip("/")
QUANT   = sys.argv[2] if len(sys.argv) > 2 else "q4_k_m"
OUT_DIR = str(Path(ADAPTER).parent / "gguf")

print(f"loading {ADAPTER}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = ADAPTER,
    max_seq_length  = 2048,
    dtype           = None,
    load_in_4bit    = True,
)

print(f"exporting to {OUT_DIR}  (quant={QUANT})")
model.save_pretrained_gguf(
    OUT_DIR,
    tokenizer,
    quantization_method = QUANT,
)
print(f"\ndone — files in {OUT_DIR}")
