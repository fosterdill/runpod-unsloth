"""
Interactive chat REPL for any trained adapter (or any HF model).

Usage:
    python train/chat.py /workspace/runs/my-run/adapter
    python train/chat.py /workspace/runs/my-run/adapter --format chat --system "You are a pirate."
    python train/chat.py unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --format chat

Two modes:
  --format alpaca  (default)  Single-turn. Each user message is sent to the model as a
                              fresh Alpaca-style instruction. Use this when your adapter
                              was trained on Alpaca-formatted data (the smoke test).
                              History is kept for /save but doesn't affect generation.

  --format chat               Multi-turn. Uses tokenizer.apply_chat_template to build
                              a prompt from the full conversation. Use this when your
                              base model is *-Instruct or your adapter was trained on
                              chat-formatted data (messages with role/content).

Slash commands while chatting:
  /help              show commands
  /reset             clear conversation history
  /system <text>     set/update the system prompt
  /temp <0..2>       set sampling temperature (default 0.7)
  /tokens <N>        set max new tokens per response (default 512)
  /save <path>       write conversation to a file
  /exit, /quit       leave
"""
import argparse
import sys
import readline  # noqa: F401  enables ↑/↓ history in input()

from unsloth import FastLanguageModel
from transformers import TextStreamer

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def alpaca_prompt(message: str, system: str = "") -> str:
    """Single-turn Alpaca format."""
    instruction = (system + "\n\n" + message) if system else message
    return ALPACA_TEMPLATE.format(instruction=instruction, input="")


def chat_prompt(history, tokenizer, system: str = "") -> str:
    """Multi-turn via the model's tokenizer chat template."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(history)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def print_help():
    print("""
commands:
  /help              show this help
  /reset             clear conversation history
  /system <text>     set the system prompt
  /temp <0..2>       set sampling temperature
  /tokens <N>        set max new tokens per response
  /save [path]       save the conversation to a file (default /workspace/chat.log)
  /exit, /quit       leave
""")


def main():
    p = argparse.ArgumentParser(
        description="Interactive chat with a trained adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("adapter", help="Path to adapter dir (or any HF model name)")
    p.add_argument("--format", default="alpaca", choices=["alpaca", "chat"],
                   help="alpaca: single-turn (default). chat: multi-turn via chat template.")
    p.add_argument("--system", default="", help="Initial system prompt")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-seq-length", type=int, default=2048)
    args = p.parse_args()

    print(f"loading {args.adapter} ...", file=sys.stderr)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.adapter,
        max_seq_length  = args.max_seq_length,
        dtype           = None,
        load_in_4bit    = True,
    )
    FastLanguageModel.for_inference(model)

    if args.format == "chat" and not getattr(tokenizer, "chat_template", None):
        sys.exit(
            f"\nerror: {args.adapter} has no tokenizer.chat_template, so --format chat won't work.\n"
            "this is normal for base (non-Instruct) models. either:\n"
            "  - load the -Instruct variant of this model, or\n"
            "  - rerun with --format alpaca (single-turn).\n"
        )

    print(f"\n  chat mode: {args.format}")
    print(f"  format=alpaca = single-turn  |  format=chat = multi-turn")
    print(f"  /help for commands, /exit to quit\n")

    history = []
    state = {
        "system": args.system,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    while True:
        try:
            line = input("\n\033[1mYou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        # --- slash commands ---
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/help":
                print_help(); continue
            if cmd == "/reset":
                history.clear()
                print("[history cleared]")
                continue
            if cmd == "/system":
                state["system"] = arg
                print(f"[system prompt set to: {arg!r}]")
                continue
            if cmd == "/temp":
                try:
                    state["temperature"] = float(arg)
                    print(f"[temperature: {state['temperature']}]")
                except ValueError:
                    print("usage: /temp 0.7")
                continue
            if cmd == "/tokens":
                try:
                    state["max_tokens"] = int(arg)
                    print(f"[max_tokens: {state['max_tokens']}]")
                except ValueError:
                    print("usage: /tokens 512")
                continue
            if cmd == "/save":
                path = arg or "/workspace/chat.log"
                with open(path, "w") as f:
                    if state["system"]:
                        f.write(f"[system]\n{state['system']}\n\n")
                    for m in history:
                        f.write(f"[{m['role']}]\n{m['content']}\n\n")
                print(f"[saved {len(history)} messages to {path}]")
                continue
            print(f"unknown command: {cmd}. /help for list.")
            continue

        # --- generate response ---
        history.append({"role": "user", "content": line})
        if args.format == "alpaca":
            prompt = alpaca_prompt(line, state["system"])
        else:
            prompt = chat_prompt(history, tokenizer, state["system"])

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        print("\033[1mBot>\033[0m ", end="", flush=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        try:
            out = model.generate(
                **inputs,
                streamer            = streamer,
                max_new_tokens      = state["max_tokens"],
                do_sample           = True,
                temperature         = state["temperature"],
                top_p               = args.top_p,
                use_cache           = True,
                repetition_penalty  = 1.1,
                pad_token_id        = tokenizer.eos_token_id,
            )
        except KeyboardInterrupt:
            print("\n[generation interrupted]")
            history.pop()  # don't keep the unanswered turn in history
            continue

        # Capture text for history (the streamer already printed it).
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": response})

    print("bye")


if __name__ == "__main__":
    main()
