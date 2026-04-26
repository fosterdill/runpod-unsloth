#!/usr/bin/env python3
"""
pod.py — one-file CLI for spinning RunPod Secure Cloud pods up and down with
your Unsloth + wandb stack and a persistent network volume attached.

Usage:
  ./pod.py up                  # create a pod, attach volume, write ssh alias
  ./pod.py status              # list your active pods
  ./pod.py ssh   [POD_ID]      # exec into the most recent (or named) pod
  ./pod.py code  [POD_ID]      # open VS Code Remote-SSH into /workspace
  ./pod.py jupyter [POD_ID]    # open Jupyter Lab in your browser
  ./pod.py logs  [POD_ID]      # print pod logs (best-effort via runpodctl)
  ./pod.py stop  [POD_ID]      # halt pod (resumable, container disk billed)
  ./pod.py down  [POD_ID]      # terminate pod (cheapest; volume data persists)
  ./pod.py gpus                # list GPU type ids RunPod is currently exposing
  ./pod.py volumes             # show your network volumes (datacenter ids)
  ./pod.py info  [POD_ID]      # full pod metadata as JSON

If POD_ID is omitted, pod.py uses the last pod id it created
(stored in .last_pod next to this script).

`up` also writes a `Host runpod` entry into ~/.ssh/config (between marked
sentinels, idempotent), so `ssh runpod` and VS Code Remote-SSH "runpod"
always point at the live pod even though IP/port change every session.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# --- deps --------------------------------------------------------------------
try:
    import runpod
except ImportError:
    sys.exit("missing dep: pip install -r requirements.txt")

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("missing dep: pip install -r requirements.txt")

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

# --- paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"
CONFIG_FILE = HERE / "config.toml"
LAST_POD_FILE = HERE / ".last_pod"

load_dotenv(ENV_FILE)


# --- helpers -----------------------------------------------------------------
def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        die(f"missing {CONFIG_FILE}; copy config.toml.example or see README")
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def init_runpod():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        die("RUNPOD_API_KEY not set (put it in .env)")
    runpod.api_key = key


def find_public_key() -> Optional[str]:
    """Prefer PUBLIC_KEY env var, then ~/.ssh/id_ed25519.pub, then id_rsa.pub."""
    if os.environ.get("PUBLIC_KEY"):
        return os.environ["PUBLIC_KEY"]
    for fname in ("id_ed25519.pub", "id_rsa.pub"):
        p = Path.home() / ".ssh" / fname
        if p.exists():
            return p.read_text().strip()
    return None


def remember(pod_id: str):
    LAST_POD_FILE.write_text(pod_id)


def recall() -> Optional[str]:
    if LAST_POD_FILE.exists():
        return LAST_POD_FILE.read_text().strip() or None
    return None


def resolve_pod_id(arg: Optional[str]) -> str:
    if arg:
        return arg
    last = recall()
    if not last:
        die("no pod id given and no .last_pod recorded; run `pod.py up` first")
    return last  # type: ignore[return-value]


def auto_datacenter(volume_id: str) -> Optional[str]:
    """Look up which datacenter the network volume lives in."""
    # The Python SDK doesn't expose a dedicated `get_network_volumes` helper,
    # so we hit the GraphQL endpoint directly via runpod.api.queries.
    import requests

    api_key = runpod.api_key
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    query = """query { myself { networkVolumes { id name dataCenterId size } } }"""
    r = requests.post("https://api.runpod.io/graphql", json={"query": query}, headers=headers, timeout=30)
    r.raise_for_status()
    vols = (r.json().get("data") or {}).get("myself", {}).get("networkVolumes") or []
    for v in vols:
        if v["id"] == volume_id:
            return v.get("dataCenterId")
    return None


# --- ssh config block --------------------------------------------------------
SSH_CONFIG = Path.home() / ".ssh" / "config"
SSH_BLOCK_BEGIN = "# >>> runpod-unsloth (managed by pod.py) >>>"
SSH_BLOCK_END = "# <<< runpod-unsloth <<<"
SSH_HOST_ALIAS = "runpod"


def _read_ssh_config() -> str:
    if not SSH_CONFIG.exists():
        return ""
    return SSH_CONFIG.read_text()


def _strip_block(text: str) -> str:
    """Remove our managed block (and any trailing blank line) if present."""
    if SSH_BLOCK_BEGIN not in text:
        return text
    pre, _, rest = text.partition(SSH_BLOCK_BEGIN)
    _, _, post = rest.partition(SSH_BLOCK_END)
    out = pre.rstrip() + "\n" + post.lstrip()
    return out.strip() + "\n"


def get_pod_full(pod_id: str) -> dict:
    """Fetch the pod with the extra fields the stock SDK query omits.

    Specifically we need `machine.podHostId`, which is the username RunPod's
    proxy SSH (ssh.runpod.io) expects (e.g. `giw8xruy2o54dw-64411de3`).
    """
    import requests
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {runpod.api_key}"}
    query = """
    query getPod($podId: String!) {
      pod(input: {podId: $podId}) {
        id name desiredStatus machineId
        runtime { uptimeInSeconds ports { ip privatePort publicPort isIpPublic type } }
        machine { gpuDisplayName podHostId secureCloud dataCenterId }
      }
    }"""
    r = requests.post(
        "https://api.runpod.io/graphql",
        json={"query": query, "variables": {"podId": pod_id}},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(f"graphql error: {body['errors']}")
    return body.get("data", {}).get("pod") or {}


def find_identity_file() -> str:
    """Return the path to the user's preferred SSH private key for the alias."""
    for fname in ("id_ed25519", "id_rsa"):
        p = Path.home() / ".ssh" / fname
        if p.exists():
            return str(p).replace(str(Path.home()), "~", 1)
    return "~/.ssh/id_ed25519"  # default; SSH will fall back to agent if absent


# SSH login user inside the pod. The unsloth/unsloth image runs as a
# non-root user `unsloth` (uid 1001) whose home directory is /workspace —
# the network volume. This is exactly what we want: dropping our public
# key into /workspace/.ssh/authorized_keys means the key persists across
# every pod that mounts this volume.
POD_SSH_USER = "unsloth"
POD_AUTHORIZED_KEYS = "/workspace/.ssh/authorized_keys"


def write_ssh_alias(ip: str, port: int):
    """Insert/replace a Host runpod block using DIRECT SSH (unsloth@ip:port).

    Direct SSH supports PTY (so VS Code Remote-SSH works), unlike RunPod's
    proxy SSH which doesn't. To make this work the pod's authorized_keys
    has to contain your public key — that's what bootstrap_authorized_keys()
    does, using the proxy SSH path (which lands as the same `unsloth` user)
    as the channel.
    """
    SSH_CONFIG.parent.mkdir(mode=0o700, exist_ok=True)
    identity = find_identity_file()
    block = (
        f"{SSH_BLOCK_BEGIN}\n"
        f"Host {SSH_HOST_ALIAS}\n"
        f"  HostName {ip}\n"
        f"  Port {port}\n"
        f"  User {POD_SSH_USER}\n"
        f"  IdentityFile {identity}\n"
        f"  IdentitiesOnly yes\n"
        f"  StrictHostKeyChecking no\n"
        f"  UserKnownHostsFile /dev/null\n"
        f"  LogLevel ERROR\n"
        f"  ServerAliveInterval 60\n"
        f"  ServerAliveCountMax 3\n"
        f"{SSH_BLOCK_END}\n"
    )
    cur = _strip_block(_read_ssh_config())
    new = (cur.rstrip() + "\n\n" + block) if cur.strip() else block
    SSH_CONFIG.write_text(new)
    SSH_CONFIG.chmod(0o600)
    print(f"[ssh-config] {SSH_CONFIG}: alias `{SSH_HOST_ALIAS}` -> "
          f"{POD_SSH_USER}@{ip}:{port}  (identity: {identity})")


def bootstrap_authorized_keys(pod_host_id: str, pubkey: str) -> tuple[bool, str]:
    """Append `pubkey` to /workspace/.ssh/authorized_keys via proxy SSH.

    Why /workspace and not /root: the unsloth/unsloth image runs as user
    `unsloth` (uid 1001) with home dir /workspace. Proxy SSH lands you as
    that same user. So we can write into /workspace/.ssh/authorized_keys
    (which we have permission for) and direct SSH as user `unsloth` on the
    pod's exposed port 22 will accept the key.

    BIG bonus: /workspace is the network volume, so once the key is there,
    every future pod that mounts this volume sees it immediately — the
    bootstrap is essentially a no-op forever after the first time.

    Implementation notes:
      * `-tt` forces client-side PTY allocation. RunPod's proxy SSH refuses
        sessions where the client won't accept a PTY ("Your SSH client
        doesn't support PTY").
      * We don't pass a remote command — the proxy ignores command args and
        only gives you an interactive shell. So we drive that shell over
        stdin (script + `exit`), and base64-embed the pubkey to avoid any
        PTY input quoting weirdness.

    Returns (success, message). Idempotent — appends only if not present.
    """
    import base64
    if not pubkey:
        return False, "no local public key found (~/.ssh/id_ed25519.pub or id_rsa.pub)"
    identity = os.path.expanduser(find_identity_file())
    encoded = base64.b64encode(pubkey.encode()).decode()
    ak = POD_AUTHORIZED_KEYS

    script = (
        "set -e\n"
        f"mkdir -p $(dirname {ak}) && chmod 700 $(dirname {ak})\n"
        f"touch {ak}\n"
        f"K=$(echo {encoded} | base64 -d)\n"
        f"grep -qxF \"$K\" {ak} || echo \"$K\" >> {ak}\n"
        f"chmod 600 {ak}\n"
        "echo BOOTSTRAP_OK\n"
        "exit\n"
    )
    cmd = [
        "ssh",
        "-tt",                                  # force PTY (proxy requires it)
        "-i", identity,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=15",
        f"{pod_host_id}@ssh.runpod.io",
        # NB: no command argument; we drive the interactive shell via stdin.
    ]
    try:
        r = subprocess.run(cmd, input=script,
                           capture_output=True, text=True, timeout=45)
    except FileNotFoundError:
        return False, "`ssh` not on PATH"
    except subprocess.TimeoutExpired:
        return False, ("proxy SSH bootstrap timed out (45s) — "
                       "use the RunPod web terminal to inspect")
    out = r.stdout + r.stderr
    if "BOOTSTRAP_OK" in out:
        return True, "key installed"
    # Strip ANSI/control-char noise PTY introduces, then surface what we got.
    import re
    cleaned = re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\r", "", out).strip()
    return False, cleaned[-300:] if cleaned else f"ssh exit {r.returncode}"


def remove_ssh_alias():
    text = _read_ssh_config()
    if SSH_BLOCK_BEGIN in text:
        SSH_CONFIG.write_text(_strip_block(text))
        print(f"[ssh-config] removed `{SSH_HOST_ALIAS}` alias")


def list_volumes():
    import requests

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {runpod.api_key}"}
    query = """query { myself { networkVolumes { id name dataCenterId size } } }"""
    r = requests.post("https://api.runpod.io/graphql", json={"query": query}, headers=headers, timeout=30)
    r.raise_for_status()
    return (r.json().get("data") or {}).get("myself", {}).get("networkVolumes") or []


# --- commands ----------------------------------------------------------------
def cmd_up(args, cfg):
    if not cfg["volume"]["network_volume_id"]:
        die("config.toml -> [volume].network_volume_id is empty.\n"
            "  1. Create a network volume in the RunPod UI (Storage tab),\n"
            "     pick a datacenter that has 4090s (e.g. EU-RO-1, US-CA-2),\n"
            "  2. Paste its id into config.toml.")

    dc = cfg["cloud"].get("data_center_id") or auto_datacenter(cfg["volume"]["network_volume_id"])
    if not dc:
        die("could not resolve datacenter for that network volume; set [cloud].data_center_id manually")

    # Build env: bake-in non-secret vars from config + secrets from .env
    env = dict(cfg.get("env") or {})
    for var in ("WANDB_API_KEY", "HF_TOKEN", "WANDB_PROJECT", "WANDB_ENTITY", "JUPYTER_PASSWORD"):
        if os.environ.get(var):
            env[var] = os.environ[var]

    pubkey = find_public_key()
    if pubkey:
        env["PUBLIC_KEY"] = pubkey
    else:
        print("[warn] no SSH public key found locally and PUBLIC_KEY not set in .env; "
              "you'll be limited to RunPod's web SSH.", file=sys.stderr)

    name = f"{cfg['pod']['name_prefix']}-{int(time.time())}"
    image = args.image or cfg["image"]["name"]
    gpu_type = args.gpu or cfg["gpu"]["type_id"]

    print(f"[up] gpu={gpu_type}  cloud={cfg['cloud']['type']}  dc={dc}")
    print(f"[up] image={image}")
    print(f"[up] volume={cfg['volume']['network_volume_id']} -> {cfg['volume']['mount_path']}")

    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=cfg["gpu"]["count"],
        cloud_type=cfg["cloud"]["type"],
        data_center_id=dc,
        support_public_ip=True,
        start_ssh=True,
        container_disk_in_gb=cfg["pod"]["container_disk_in_gb"],
        min_vcpu_count=cfg["pod"]["min_vcpu_count"],
        min_memory_in_gb=cfg["pod"]["min_memory_in_gb"],
        ports=cfg["pod"]["ports"],
        volume_mount_path=cfg["volume"]["mount_path"],
        env=env,
        network_volume_id=cfg["volume"]["network_volume_id"],
    )

    pod_id = pod["id"]
    remember(pod_id)
    print(f"[up] pod {pod_id} created. polling for SSH (up to {args.timeout}s)...")

    if connect_and_persist(pod_id, timeout=args.timeout):
        print_connect_help(pod_id)
    else:
        print()
        print(f"[up] pod {pod_id} not RUNNING within {args.timeout}s. The pod may")
        print("     still be pulling its image — large images take 5-10 min on first boot.")
        print()
        print("  ./pod.py wait                # keep polling and write the alias when ready")
        print("  ./pod.py status              # see if the pod is RUNNING")
        print("  ./pod.py logs                # peek at boot logs")


def wait_for_ssh(pod_id: str, timeout: int = 600, quiet: bool = False):
    """Poll until pod is RUNNING, podHostId is set, and port 22 is exposed.

    Returns dict {pod_host_id, ip, port} on success, None on timeout.
    Prints state transitions while waiting.
    """
    deadline = time.time() + timeout
    last_state = None
    started = time.time()
    while time.time() < deadline:
        try:
            info = get_pod_full(pod_id)
        except Exception as e:
            if not quiet:
                print(f"[wait] transient API error: {e}; retrying", file=sys.stderr)
            time.sleep(5)
            continue
        state = info.get("desiredStatus") or "?"
        machine = info.get("machine") or {}
        host_id = machine.get("podHostId") or ""
        runtime = info.get("runtime") or {}

        # Look for the public port-22 mapping (needed for direct SSH / VS Code).
        ssh_port = None
        ssh_ip = None
        for p in runtime.get("ports") or []:
            if p.get("privatePort") == 22 and p.get("isIpPublic"):
                ssh_ip = p["ip"]; ssh_port = p["publicPort"]; break

        if not quiet and state != last_state:
            elapsed = int(time.time() - started)
            print(f"[wait] +{elapsed:>3}s  state={state}  "
                  f"uptime={runtime.get('uptimeInSeconds', 0)}s  "
                  f"podHostId={host_id or '(pending)'}  "
                  f"ssh22={'(pending)' if not ssh_port else f'{ssh_ip}:{ssh_port}'}")
            last_state = state
        if state == "RUNNING" and host_id and ssh_port:
            return {"pod_host_id": host_id, "ip": ssh_ip, "port": ssh_port}
        time.sleep(5)
    return None


def connect_and_persist(pod_id: str, timeout: int = 600):
    """Wait for ready, bootstrap authorized_keys, write direct-SSH alias.

    Two-step because:
      1. RunPod's proxy SSH (ssh.runpod.io) authenticates with our account
         key but doesn't support PTY → bad for VS Code Remote-SSH.
      2. Direct SSH (root@ip:port) supports PTY → good for VS Code, but the
         pod's authorized_keys must contain our pubkey for it to authenticate.
    So: use proxy SSH to install our pubkey into authorized_keys (bootstrap),
    then point the alias at direct SSH for everything else.

    Returns the endpoint dict on success, None on timeout.
    """
    target = wait_for_ssh(pod_id, timeout=timeout)
    if not target:
        return None

    pubkey = find_public_key()
    if not pubkey:
        print("[bootstrap] no local public key (~/.ssh/id_ed25519.pub or id_rsa.pub); "
              "skipping authorized_keys install. `ssh runpod` will fail until you "
              "add a key. Generate one with: ssh-keygen -t ed25519", file=sys.stderr)
    else:
        ok, msg = bootstrap_authorized_keys(target["pod_host_id"], pubkey)
        if ok:
            print(f"[bootstrap] {msg}: ensured pubkey is in {POD_AUTHORIZED_KEYS} "
                  f"(persists on the network volume)")
        else:
            print(f"[bootstrap] WARN: {msg}", file=sys.stderr)
            print(f"[bootstrap] direct SSH will fail. As a workaround, ssh into the "
                  f"pod via proxy ({target['pod_host_id']}@ssh.runpod.io) and append "
                  f"your pubkey to {POD_AUTHORIZED_KEYS} manually.", file=sys.stderr)

    write_ssh_alias(target["ip"], target["port"])
    return target


def print_connect_help(pod_id: str):
    print()
    print("  ssh runpod                   # direct SSH via ~/.ssh/config alias")
    print("  ./pod.py code                # open VS Code Remote-SSH on /workspace")
    print(f"  ./pod.py jupyter             # https://{pod_id}-8888.proxy.runpod.net")


def cmd_status(args, cfg):
    pods = runpod.get_pods()
    if not pods:
        print("no active pods.")
        return
    rows = []
    for p in pods:
        rows.append((
            p.get("id", "?")[:12],
            p.get("name", "")[:24],
            p.get("desiredStatus", ""),
            (p.get("machine") or {}).get("gpuDisplayName", "")[:24],
            f"${p.get('costPerHr', 0):.2f}/hr",
        ))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(r[i].ljust(w[i]) for i in range(len(r))))


def cmd_ssh(args, cfg):
    pod_id = resolve_pod_id(args.pod_id)
    if not connect_and_persist(pod_id, timeout=args.timeout):
        die(f"pod {pod_id} not RUNNING after {args.timeout}s. "
            f"Try `./pod.py wait` or check `./pod.py status`.")
    cmd = ["ssh", SSH_HOST_ALIAS]
    print("$", " ".join(cmd))
    os.execvp("ssh", cmd)


def cmd_code(args, cfg):
    """Open VS Code Remote-SSH into /workspace/runpod-unsloth on the pod."""
    pod_id = resolve_pod_id(args.pod_id)
    if not connect_and_persist(pod_id, timeout=args.timeout):
        die(f"pod {pod_id} not RUNNING after {args.timeout}s. "
            f"Try `./pod.py wait` first.")
    if not shutil.which("code"):
        die("`code` CLI not on PATH. In VS Code: Cmd+Shift+P -> "
            "'Shell Command: Install code command in PATH'.")
    folder = args.folder or "/workspace/runpod-unsloth"
    uri = f"vscode-remote://ssh-remote+{SSH_HOST_ALIAS}{folder}"
    print(f"[code] opening {uri}")
    subprocess.run(["code", "--folder-uri", uri], check=False)


def cmd_jupyter(args, cfg):
    """Open RunPod's HTTPS proxy URL for the pod's Jupyter server."""
    import webbrowser
    pod_id = resolve_pod_id(args.pod_id)
    url = f"https://{pod_id}-8888.proxy.runpod.net/lab"
    print(f"[jupyter] {url}")
    if not args.print_only:
        webbrowser.open(url)


def cmd_wait(args, cfg):
    """Poll an existing pod until it's RUNNING, then write the SSH alias.

    Useful when `up` timed out but the pod is still booting (e.g. pulling a
    large image). Idempotent — safe to run multiple times.
    """
    pod_id = resolve_pod_id(args.pod_id)
    print(f"[wait] polling pod {pod_id} for up to {args.timeout}s")
    if not connect_and_persist(pod_id, timeout=args.timeout):
        die(f"pod {pod_id} still not RUNNING after {args.timeout}s. "
            f"Check ./pod.py status — it may have failed to start.")
    print_connect_help(pod_id)


def cmd_logs(args, cfg):
    pod_id = resolve_pod_id(args.pod_id)
    runpodctl = shutil.which("runpodctl")
    if runpodctl:
        subprocess.run([runpodctl, "get", "pod", pod_id, "--logs"], check=False)
    else:
        info = runpod.get_pod(pod_id)
        print(json.dumps(info, indent=2))
        print("\n(install runpodctl for streaming logs: https://github.com/runpod/runpodctl)")


def cmd_stop(args, cfg):
    pod_id = resolve_pod_id(args.pod_id)
    print(f"[stop] {pod_id}")
    print(json.dumps(runpod.stop_pod(pod_id), indent=2))


def cmd_down(args, cfg):
    pod_id = resolve_pod_id(args.pod_id)
    if not args.yes:
        confirm = input(f"terminate pod {pod_id}? container disk will be wiped, "
                        f"network volume is safe. [y/N] ").strip().lower()
        if confirm != "y":
            print("aborted.")
            return
    print(f"[down] {pod_id}")
    print(json.dumps(runpod.terminate_pod(pod_id), indent=2))
    if recall() == pod_id:
        LAST_POD_FILE.unlink(missing_ok=True)
    remove_ssh_alias()


def cmd_gpus(args, cfg):
    gpus = runpod.get_gpus()
    for g in gpus:
        if args.filter.lower() in g.get("displayName", "").lower():
            print(f"  {g.get('id', ''):44}  {g.get('displayName', '')}  ({g.get('memoryInGb', '?')}GB)")


def cmd_volumes(args, cfg):
    vols = list_volumes()
    if not vols:
        print("no network volumes. create one in the RunPod UI -> Storage.")
        return
    for v in vols:
        print(f"  {v['id']}  {v['name']:24}  dc={v.get('dataCenterId','?')}  size={v.get('size','?')}GB")


def cmd_info(args, cfg):
    pod_id = resolve_pod_id(args.pod_id)
    print(json.dumps(runpod.get_pod(pod_id), indent=2))


# --- arg parsing -------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("up", help="spin up a pod")
    sp.add_argument("--gpu", help="override config.toml [gpu].type_id")
    sp.add_argument("--image", help="override config.toml [image].name")
    sp.add_argument("--timeout", type=int, default=600,
                    help="seconds to wait for SSH after creating pod (default 600)")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("status", help="list active pods")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("wait", help="poll for SSH on an existing pod and write alias")
    sp.add_argument("pod_id", nargs="?")
    sp.add_argument("--timeout", type=int, default=900, help="seconds to wait (default 900)")
    sp.set_defaults(func=cmd_wait)

    sp = sub.add_parser("ssh", help="ssh into a pod (default: last created)")
    sp.add_argument("pod_id", nargs="?")
    sp.add_argument("--timeout", type=int, default=120, help="seconds to wait for SSH (default 120)")
    sp.set_defaults(func=cmd_ssh)

    sp = sub.add_parser("code", help="open VS Code Remote-SSH on /workspace/runpod-unsloth")
    sp.add_argument("pod_id", nargs="?")
    sp.add_argument("--folder", help="path on the pod to open (default /workspace/runpod-unsloth)")
    sp.add_argument("--timeout", type=int, default=120, help="seconds to wait for SSH (default 120)")
    sp.set_defaults(func=cmd_code)

    sp = sub.add_parser("jupyter", help="open RunPod-proxied Jupyter URL in browser")
    sp.add_argument("pod_id", nargs="?")
    sp.add_argument("--print-only", action="store_true", help="just print the URL")
    sp.set_defaults(func=cmd_jupyter)

    sp = sub.add_parser("logs", help="show pod logs")
    sp.add_argument("pod_id", nargs="?")
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("stop", help="halt pod (resumable; container disk still billed)")
    sp.add_argument("pod_id", nargs="?")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("down", help="terminate pod (cheapest; network volume persists)")
    sp.add_argument("pod_id", nargs="?")
    sp.add_argument("-y", "--yes", action="store_true", help="skip confirmation")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("gpus", help="list available GPU types")
    sp.add_argument("--filter", default="", help="substring filter, e.g. '4090'")
    sp.set_defaults(func=cmd_gpus)

    sp = sub.add_parser("volumes", help="list your network volumes")
    sp.set_defaults(func=cmd_volumes)

    sp = sub.add_parser("info", help="dump full pod metadata as JSON")
    sp.add_argument("pod_id", nargs="?")
    sp.set_defaults(func=cmd_info)

    args = p.parse_args()
    cfg = load_config()
    init_runpod()
    args.func(args, cfg)


if __name__ == "__main__":
    main()
