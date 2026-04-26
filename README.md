# runpod-unsloth

Ergonomic RunPod Secure Cloud setup for Unsloth fine-tuning of 4B–9B models on
a single 4090, with wandb logging and a persistent network volume so models
and datasets stay alive between sessions.

```
runpod-unsloth/
├── Dockerfile           # FROM unsloth/unsloth + wandb + entrypoint (optional)
├── entrypoint.sh        # only used if you build the custom image
├── pod.py               # local CLI: up / down / ssh / code / status / volumes / gpus
├── config.toml          # GPU, image, volume, ports — edit once
├── .env.example         # secrets (RunPod + wandb + HF tokens)
├── requirements.txt     # for pod.py (runpod, python-dotenv, tomli)
└── train/               # runs ON the pod
    ├── train.py            # generic trainer, takes a YAML config
    ├── chat.py             # interactive chat REPL for any adapter
    ├── export_gguf.py      # merge LoRA + quantize → llama.cpp-ready file
    └── configs/
        ├── alpaca-smoke.yaml             # 60-step smoke test on Alpaca
        ├── llama3-8b-instruct-chat.yaml  # multi-turn chat on Ultrachat
        ├── jsonl-local.yaml              # train on your own JSONL
        └── README.md                     # full config schema
```

## Training a new model

The whole point of the `train/` setup: each fine-tune is one YAML config file.

```bash
# On the pod (after ./pod.py code or ssh runpod)
cd /workspace/runpod-unsloth

# Run any config:
python train/train.py train/configs/alpaca-smoke.yaml

# Or override fields from the CLI without editing the file:
python train/train.py train/configs/alpaca-smoke.yaml \
    --name alpaca-1k --max-steps 1000 --lr 1e-4
```

Add a new training run by dropping a new `.yaml` in `train/configs/`. See
`train/configs/README.md` for the full field list — only `name`, `model`,
`data.source`, and `data.format` are required.

## Chatting with a trained adapter

```bash
# Single-turn (Alpaca-trained adapters)
python train/chat.py /workspace/runs/alpaca-smoke/adapter

# Multi-turn (chat-trained adapters or *-Instruct base models)
python train/chat.py /workspace/runs/my-chat-run/adapter --format chat \
    --system "You are a terse senior engineer."
```

Slash commands work mid-conversation: `/reset`, `/system <text>`, `/temp 0.3`,
`/save chat.log`, `/exit`. Up-arrow recalls previous prompts.

## Exporting for laptop inference (llama.cpp)

```bash
# On the pod
python train/export_gguf.py /workspace/runs/alpaca-smoke/adapter         # default q4_k_m
python train/export_gguf.py /workspace/runs/alpaca-smoke/adapter q5_k_m  # better quality

# On your laptop
scp 'runpod:/workspace/runs/alpaca-smoke/gguf/*.gguf' ~/models/
brew install llama.cpp
llama-server -m ~/models/unsloth.Q4_K_M.gguf -c 4096
# open http://localhost:8080
```

## One-time setup

1. **Install local deps** (for `pod.py`).
   ```bash
   cd runpod-unsloth
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get your RunPod API key.** Account → *Settings* → *API Keys* → *Create*.
   Copy `.env.example` to `.env` and paste it in. Add your `WANDB_API_KEY`
   while you're there (Weights & Biases → *User settings* → *API keys*).

3. **Create the network volume.** RunPod web UI → *Storage* → *Network
   Volumes* → *New*. Pick a datacenter that has 4090s on Secure Cloud
   (EU-RO-1, US-CA-2, US-KS-2, and CA-MTL-1 all reliably do). 200 GB is a
   solid starting size for a few base models + checkpoints — you can grow
   it later. Standard storage tier is fine.

4. **Paste the volume id** into `config.toml` under `[volume].network_volume_id`.
   Sanity-check with:
   ```bash
   ./pod.py volumes
   ```
   pod.py auto-detects the datacenter from the volume; you don't need to set
   `[cloud].data_center_id` unless you want to pin it.

5. **Build and push the image** (skip this step if you set `[image].name` to
   `unsloth/unsloth:latest` in config.toml — you lose the `WANDB_DIR` /
   `HF_HOME` defaults but pod.py still injects WANDB_API_KEY).
   ```bash
   export DOCKER_USER=fosterdill           # your Docker Hub user
   docker build --platform=linux/amd64 -t $DOCKER_USER/runpod-unsloth:latest .
   docker push $DOCKER_USER/runpod-unsloth:latest
   ```
   Then update `[image].name` in `config.toml` to
   `fosterdill/runpod-unsloth:latest`.

6. **Add an SSH key.** pod.py picks up `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`)
   automatically. If you'd rather use a different key, set `PUBLIC_KEY=...`
   in `.env`.

## Daily workflow

```bash
./pod.py up                  # ~30-90 s; writes `Host runpod` to ~/.ssh/config
./pod.py code                # opens VS Code Remote-SSH into /workspace/runpod-unsloth
                             # ...edit, run training in VS Code's terminal...
./pod.py down -y             # terminate; only the volume (~$14/mo) keeps billing
```

That's the loop. Three commands. Below is what each does and the
alternatives if you don't use VS Code.

### `./pod.py up`

Creates a new Secure Cloud pod with your network volume mounted at
`/workspace`, then writes a managed block into `~/.ssh/config`:

```sshconfig
# >>> runpod-unsloth (managed by pod.py) >>>
Host runpod
  HostName 213.x.x.x
  Port 12345
  User root
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 60
# <<< runpod-unsloth <<<
```

That alias means **`ssh runpod` always works** without you typing
IP/port, and VS Code's "Connect to Host… → runpod" just connects.

### `./pod.py code`

Resolves the live pod's IP/port, refreshes the SSH alias, then runs
`code --folder-uri vscode-remote://ssh-remote+runpod/workspace/runpod-unsloth`.
Prereqs: install the **Remote - SSH** extension in VS Code and put the
`code` CLI on your PATH (Cmd+Shift+P → "Shell Command: Install 'code'
command in PATH").

VS Code's integrated terminal lands on the pod in `/workspace`. Run
training there inside `tmux` so disconnects don't kill the job:

```bash
tmux new -s train
python /workspace/runpod-unsloth/train/example_qlora.py
# detach: Ctrl+b then d        reattach: tmux attach -t train
```

### `./pod.py jupyter`

Opens `https://<pod-id>-8888.proxy.runpod.net/lab` in your default
browser. RunPod proxies Jupyter through their auth, so the port doesn't
need to be public. Use it for ad-hoc notebook poking; use VS Code for
real editing.

### `./pod.py down -y`

Terminates the pod and removes the SSH alias from `~/.ssh/config`. The
network volume is untouched, so everything in `/workspace/...`
(checkpoints, datasets, HF cache, wandb cache, the cloned repo itself)
is waiting next time.

### Without VS Code

```bash
./pod.py up
ssh runpod                          # land at /workspace
cd runpod-unsloth                   # the cloned repo lives on the volume
tmux new -s train
python train/example_qlora.py
# Ctrl+b d to detach, exit to drop the SSH session
./pod.py down -y                    # alias auto-removed
```

`pod.py up` always creates a *new* pod — the network volume is what
persists. `down` terminates that pod (cheapest). Use `stop` instead if
you specifically want to resume the same container disk within a day.

## Priming the network volume (one time)

The first time you bring a pod up the volume is empty. Bootstrap it once:

```bash
./pod.py up
ssh runpod
cd /workspace
git clone <your repo>            runpod-unsloth   # if it's in git, or:
mkdir runpod-unsloth && exit
# from your laptop:
rsync -avz ~/code/runpod-unsloth/ runpod:/workspace/runpod-unsloth/
```

From then on every pod boot has the repo at `/workspace/runpod-unsloth`.
Iteration loop while editing in VS Code Remote-SSH writes directly to
the network volume — no syncing needed.

## Costs to watch

* **4090 Secure Cloud:** ~$0.69/hr while running. Stops billing the second
  `down` returns.
* **Network volume:** $0.07/GB/mo standard. A 200 GB volume = ~$14/mo
  whether or not a pod is attached. This is the only ongoing cost between
  sessions.
* **Container disk** (50 GB by default in `config.toml`): only billed while
  the pod exists. `down` wipes it; `stop` keeps it and bills storage.

## What lives where

| path                              | persists?                       |
| --------------------------------- | ------------------------------- |
| `/workspace/...`                  | yes — network volume            |
| `/workspace/.cache/huggingface`   | yes — HF model cache            |
| `/workspace/runs/<run-name>`      | yes — your checkpoints, configs |
| `/workspace/wandb`                | yes — wandb local cache         |
| anywhere else (`/root`, `/tmp`)   | gone on `down`                  |

The Dockerfile pre-sets `HF_HOME`, `WANDB_DIR`, and `TRANSFORMERS_CACHE` to
land under `/workspace/.cache`, so `huggingface_hub.snapshot_download(...)`
and `wandb.init(...)` write into the volume by default. No code changes
needed.

## Sizing notes for 4B–9B on a 4090 (24 GB)

| model            | mode  | seq_len | batch x grad_accum | VRAM   |
| ---------------- | ----- | ------- | ------------------ | ------ |
| Llama-3.2-3B     | LoRA  | 4096    | 4 x 4              | ~16 GB |
| Mistral-7B       | QLoRA | 2048    | 2 x 4              | ~17 GB |
| Llama-3.1-8B     | QLoRA | 2048    | 2 x 4              | ~19 GB |
| Gemma-2-9B       | QLoRA | 1024    | 1 x 8              | ~22 GB |

Always use `use_gradient_checkpointing="unsloth"` — it's free memory.

## Troubleshooting

* **`could not resolve datacenter`** — your network volume id is wrong, or
  the API key doesn't see it. Run `./pod.py volumes` to confirm.
* **`gpu_type_id` rejected** — RunPod occasionally renames SKUs. Run
  `./pod.py gpus --filter 4090` and copy the exact `id` into `config.toml`.
* **Pod is RUNNING but `ssh` hangs** — give it ~20 s after first boot;
  sshd starts as part of the entrypoint. The RunPod web UI's *Connect → SSH*
  also works as a fallback.
* **`wandb: ERROR Network error`** inside training — RunPod outbound is fine
  by default; this is almost always a stale `WANDB_API_KEY`. Re-run
  `wandb login` inside the pod, or just `./pod.py down` and `./pod.py up`
  again with a fresh `WANDB_API_KEY` in `.env`.

## Why this layout

* **Custom Docker image, thin.** `FROM unsloth/unsloth` means we inherit
  pinned PyTorch/CUDA/xformers/bitsandbytes. We add only wandb,
  hf_transfer, tmux, and an entrypoint — no version drift.
* **Network volume for state, container for compute.** Container is
  disposable, volume isn't. `down` is reflex-cheap.
* **`WANDB_API_KEY` injected per-pod, not baked.** It rides in via the
  `env=` parameter to `runpod.create_pod`, so the image stays shareable
  and the key never lands in a docker layer.
* **CLI over UI.** Pod creation goes through `runpod.create_pod` so every
  field is reproducible from `config.toml`. No clicking through the
  deploy wizard each time.
