"""Download the HuggingFace model weights used by the CoT-DDI pipeline.

Usage
-----
    # Optional — point HF_HOME at the cache you want to populate.
    export HF_HOME="$HOME/.cache/huggingface"

    # Optional — accept the gated repo licences on the HF website, then:
    export HUGGING_FACE_HUB_TOKEN=hf_xxx

    # All models (~600 GB total — large!)
    python scripts/download_models.py

    # Just the student base
    python scripts/download_models.py --only student

    # A specific subset
    python scripts/download_models.py --only teachers prm

Models
------
Phase B teacher ensemble:
    meta-llama/Llama-3.3-70B-Instruct
    Qwen/Qwen2.5-72B-Instruct
    deepseek-ai/DeepSeek-R1-Distill-Llama-70B

Phase B DDI-PRM base (default = Med-PRM reward checkpoint):
    dmis-lab/llama-3.1-medprm-reward-v1.0
    meta-llama/Llama-3.1-8B-Instruct   (fallback)

Phase C student base:
    Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_DATASETS_OFFLINE", None)

from huggingface_hub import snapshot_download

MODELS: list[tuple[str, str]] = [
    ("meta-llama/Llama-3.3-70B-Instruct",         "teacher · llama-3.3-70b"),
    ("Qwen/Qwen2.5-72B-Instruct",                 "teacher · qwen-2.5-72b"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "teacher · deepseek-r1-distill-70b"),
    ("dmis-lab/llama-3.1-medprm-reward-v1.0",     "PRM base · default (Med-PRM)"),
    ("meta-llama/Llama-3.1-8B-Instruct",          "PRM base · fallback (train from scratch)"),
    ("Qwen/Qwen2.5-7B-Instruct",                  "student base"),
]

ALIASES: dict[str, set[str]] = {
    "teachers": {
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    },
    "prm": {
        "dmis-lab/llama-3.1-medprm-reward-v1.0",
        "meta-llama/Llama-3.1-8B-Instruct",
    },
    "student": {"Qwen/Qwen2.5-7B-Instruct"},
    "qwen7b":  {"Qwen/Qwen2.5-7B-Instruct"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Download only matching repo IDs or aliases "
            "(teachers · prm · student · qwen7b)."
        ),
    )
    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "[warn] no HUGGING_FACE_HUB_TOKEN / HF_TOKEN in env; "
            "gated repos (Llama, etc.) will fail unless you set one:\n"
            "  export HUGGING_FACE_HUB_TOKEN=hf_xxx\n",
            flush=True,
        )

    selected = MODELS
    if args.only:
        wanted: set[str] = set()
        for item in args.only:
            wanted.update(ALIASES.get(item, {item}))
        selected = [(repo, label) for repo, label in MODELS if repo in wanted]
        if not selected:
            raise SystemExit(f"no models matched --only {args.only!r}")

    for repo, label in selected:
        print(f"\n→ {repo}  ({label})", flush=True)
        try:
            path = snapshot_download(repo_id=repo, token=token)
            print(f"   cached at {path}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {exc}", flush=True)
            print("   hints:")
            print("     • gated repos: visit the HF model page, accept the licence,")
            print("       then export HUGGING_FACE_HUB_TOKEN (a Read token is enough)")
            print("     • network: try `curl -sI https://huggingface.co/` to confirm")
            print("       the machine can reach HuggingFace")


if __name__ == "__main__":
    main()
