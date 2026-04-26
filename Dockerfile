# Thin layer on top of the official Unsloth image.
# unsloth/unsloth ships with PyTorch + CUDA + Unsloth + xformers + bitsandbytes
# pre-pinned and tested. We add wandb, a few QoL CLI tools, and an entrypoint
# that wires WANDB_API_KEY (passed in via RunPod env vars) before launching.
#
# Build & push (one time, after edits):
#   export DOCKER_USER=<your-dockerhub-user>
#   docker build --platform=linux/amd64 -t $DOCKER_USER/runpod-unsloth:latest .
#   docker push $DOCKER_USER/runpod-unsloth:latest
#
# Then point pod.py at $DOCKER_USER/runpod-unsloth:latest (see config.py).

FROM unsloth/unsloth:latest

# --- QoL system packages -----------------------------------------------------
# tmux: keep training alive across SSH disconnects
# htop / nvtop: see CPU & GPU live
# git-lfs: pull/push HF model weights
# rsync: sync files in/out of the network volume cleanly
# jq, less, vim: general comfort
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux htop git-lfs rsync jq less vim ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# --- Python additions --------------------------------------------------------
# wandb is usually already in unsloth's deps; reinstall to guarantee a recent
# version. hf_transfer dramatically speeds up HuggingFace downloads.
RUN pip install --no-cache-dir --upgrade \
        wandb \
        hf_transfer \
        "huggingface_hub[cli]"

# Tell HuggingFace Hub to use hf_transfer by default (much faster downloads
# onto the network volume).
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Cache HF + wandb data on the network volume so it persists across pods.
# /workspace is where RunPod mounts the network volume by default.
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_DIR=/workspace/wandb
ENV WANDB_CACHE_DIR=/workspace/.cache/wandb
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
ENV PIP_CACHE_DIR=/workspace/.cache/pip

# --- Entrypoint --------------------------------------------------------------
# Wires WANDB_API_KEY (if RunPod injected one), prints a banner, then starts
# sshd + jupyter (matching the upstream unsloth image's behavior).
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 22 8888 8000
CMD ["/usr/local/bin/entrypoint.sh"]
