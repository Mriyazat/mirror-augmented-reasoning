"""Download the exact HF model weights this pipeline needs, into $HF_HOME.

Run on a login node (internet available) BEFORE the H100 allocation:

    source scripts/slurm/activate_env.sh
    unset HF_HUB_OFFLINE                 # allow downloads
    python scripts/slurm/download_models.py

Stored at: $SCRATCH/.cache/huggingface/hub/

Models:
  # Phase B1 teacher ensemble (rubric.teacher_models)
  - Llama-3.3-70B-Instruct                 (teacher 1: general reasoning)
  - Qwen2.5-72B-Instruct                   (teacher 2: different family + tokenizer)
  - DeepSeek-R1-Distill-Llama-70B          (teacher 3: CoT-optimized)
  # Phase B PRM base (rubric.prm_base — default is the Med-PRM reward checkpoint)
  - dmis-lab/llama-3.1-medprm-reward-v1.0  (PRM-pretrained on medical steps;
                                            recommended default; 8B Llama-3.1-
                                            Instruct already fine-tuned for
                                            Med-PRM's " +"/" -" step-reward
                                            scheme, EMNLP '25 checkpoint)
  - Llama-3.1-8B-Instruct                  (fallback PRM base; use if you want
                                            to train the PRM from scratch)
  # Phase C student base
  - Qwen2.5-7B-Instruct                    (student base)
"""
from __future__ import annotations
import argparse
import os

# Force online mode BEFORE huggingface_hub is imported — `activate_env.sh`
# sets these to "1" so compute-node runs can't accidentally hit the hub,
# but this is the download script, so online is required.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_DATASETS_OFFLINE", None)

from huggingface_hub import snapshot_download

MODELS = [
    # Teachers (B1 ensemble)
    ("meta-llama/Llama-3.3-70B-Instruct",             "teacher-1 llama-3.3-70b"),
    ("Qwen/Qwen2.5-72B-Instruct",                     "teacher-2 qwen-2.5-72b"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B",     "teacher-3 deepseek-r1-distill-70b"),
    # PRM (B3) — correct repo is dmis-lab/llama-3.1-medprm-reward-v1.0
    # (no repo called "Med-PRM-7B" exists; the paper's artifact is this one).
    ("dmis-lab/llama-3.1-medprm-reward-v1.0",         "PRM base (default; Med-PRM reward v1.0)"),
    ("meta-llama/Llama-3.1-8B-Instruct",              "PRM base (fallback; train from scratch)"),
    # Student (C1)
    ("Qwen/Qwen2.5-7B-Instruct",                      "student base (C1)"),
]

ALIASES = {
    "student": {"Qwen/Qwen2.5-7B-Instruct"},
    "phase_c": {"Qwen/Qwen2.5-7B-Instruct"},
    "qwen7b": {"Qwen/Qwen2.5-7B-Instruct"},
}

token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
if not token:
    print("[warn] no HUGGING_FACE_HUB_TOKEN / HF_TOKEN in env; gated repos "
          "(Llama, etc.) may fail. Paste a token if they do:\n"
          "  export HUGGING_FACE_HUB_TOKEN=hf_xxx\n", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--only",
    nargs="+",
    default=None,
    help=(
        "Download only matching repo IDs or aliases. Useful aliases: "
        "student, phase_c, qwen7b."
    ),
)
args = parser.parse_args()

selected = MODELS
if args.only:
    wanted: set[str] = set()
    for item in args.only:
        wanted.update(ALIASES.get(item, {item}))
    selected = [(repo, label) for repo, label in MODELS if repo in wanted]
    if not selected:
        raise SystemExit(f"no models matched --only {args.only!r}")

for repo, label in selected:
    print(f"\n→ downloading {repo}  ({label})", flush=True)
    try:
        # allow_patterns=None → fetch everything.
        # We drop the deprecated `resume_download=True` kwarg; HF always
        # resumes now, and keeping it prints a UserWarning every call.
        path = snapshot_download(repo_id=repo, token=token)
        print(f"   cached at {path}", flush=True)
    except Exception as e:
        print(f"   FAILED: {e}", flush=True)
        print("   hints:")
        print("     - gated models: visit the repo page, accept the license,")
        print("       then set HUGGING_FACE_HUB_TOKEN (a Read token is enough)")
        print("     - network: try `curl -sI https://huggingface.co/` to")
        print("       confirm the login node can reach HF")
