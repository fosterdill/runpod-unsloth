#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Entrypoint for the runpod-unsloth image.
#
# Responsibilities:
#  1. Wire WANDB_API_KEY (RunPod injects it as an env var via pod.py).
#  2. Make sure the cache dirs on /workspace exist.
#  3. Start sshd so `runpodctl ssh` and `ssh root@<ip>` both work.
#  4. Start Jupyter (matches upstream unsloth/unsloth behavior).
#  5. Tail logs so the container stays alive.
# -----------------------------------------------------------------------------

echo "==================================================================="
echo " runpod-unsloth pod starting"
echo " image:       $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2-)"
echo " gpu(s):      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
echo " network vol: /workspace ($(df -h /workspace 2>/dev/null | awk 'NR==2 {print $2" total, "$4" free"}'))"
echo "==================================================================="

# --- Cache dirs on the network volume ----------------------------------------
mkdir -p /workspace/.cache/huggingface \
         /workspace/.cache/wandb \
         /workspace/.cache/pip \
         /workspace/wandb \
         /workspace/runs \
         /workspace/datasets \
         /workspace/models

# --- wandb -------------------------------------------------------------------
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[wandb] WANDB_API_KEY present, logging in non-interactively"
  wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || \
    echo "[wandb] login failed (will fall back to env-var auth at run time)"
else
  echo "[wandb] WANDB_API_KEY not set. Either set it as a RunPod env var on"
  echo "        the pod, or run \`wandb login\` manually inside the pod."
fi

# --- HuggingFace -------------------------------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[hf] HF_TOKEN present, logging into huggingface_hub"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || true
fi

# --- SSH ---------------------------------------------------------------------
# RunPod injects the user's public key into /root/.ssh/authorized_keys via the
# PUBLIC_KEY env var (set on the account). We just need sshd running.
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
# Generate host keys if missing
ssh-keygen -A 2>/dev/null || true
/usr/sbin/sshd

# --- Jupyter -----------------------------------------------------------------
# Default to a password from JUPYTER_PASSWORD if set, else no password (the
# pod is behind RunPod's auth proxy; you connect via the RunPod web UI).
JUPYTER_TOKEN="${JUPYTER_PASSWORD:-}"
mkdir -p /workspace/notebooks
cd /workspace
nohup jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.token="$JUPYTER_TOKEN" --ServerApp.password='' \
  --ServerApp.root_dir=/workspace \
  > /workspace/jupyter.log 2>&1 &

echo "==================================================================="
echo " ready. ssh: port 22   jupyter: port 8888"
echo " workspace: /workspace  (network volume, persists between sessions)"
echo "==================================================================="

# Stay alive
tail -f /workspace/jupyter.log /dev/null
