#!/usr/bin/env python3
"""
One-shot diagnostic — dumps every field RunPod returns for your pod,
plus an introspection of the Pod GraphQL type so we can find which
field maps to the SSH suffix (the `-64411de3` part of your working
proxy SSH login).

Run:  python debug_pod.py giw8xruy2o54dw
"""
import json, os, sys, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
API = "https://api.runpod.io/graphql"
KEY = os.environ.get("RUNPOD_API_KEY")
if not KEY:
    sys.exit("RUNPOD_API_KEY missing from .env")

POD_ID = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: debug_pod.py <pod_id>")

H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

# 1. Introspect the Pod type to discover every field.
intro_q = """
query { __type(name: "Pod") { fields { name type { name kind ofType { name kind } } } } }
"""
intro = requests.post(API, json={"query": intro_q}, headers=H, timeout=30).json()
fields = (intro.get("data") or {}).get("__type", {}).get("fields") or []
print("=== Pod fields available in schema ===")
for f in fields:
    t = f["type"]
    tn = t.get("name") or (t.get("ofType") or {}).get("name") or t.get("kind")
    print(f"  {f['name']:30s}  {tn}")

# 2. Try a wide query — fields we suspect are SSH-related
wide_q = """
query {
  pod(input: {podId: "%s"}) {
    id name desiredStatus machineId podType
    runtime { uptimeInSeconds ports { ip privatePort publicPort isIpPublic type } }
    machine { gpuDisplayName podHostId secureCloud dataCenterId }
  }
}
""" % POD_ID
wide = requests.post(API, json={"query": wide_q}, headers=H, timeout=30).json()
print("\n=== pod() with extra fields ===")
print(json.dumps(wide, indent=2, default=str))

# 3. Introspect Machine too
intro_m = requests.post(API, json={"query":
    'query { __type(name: "Machine") { fields { name type { name kind ofType { name } } } } }'
}, headers=H, timeout=30).json()
mf = (intro_m.get("data") or {}).get("__type", {}).get("fields") or []
print("\n=== Machine fields available in schema ===")
for f in mf:
    t = f["type"]
    tn = t.get("name") or (t.get("ofType") or {}).get("name") or t.get("kind")
    print(f"  {f['name']:30s}  {tn}")
