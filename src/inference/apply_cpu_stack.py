"""Apply the best CPU-only rescue stack: self-consistency (if rerank
candidates present) then trace-rescue with the val-tuned conservative
hint-majority rule.

Outputs:
  <input_stem>.cpu_stack.jsonl     — rescued predictions
  prints macro-F1 deltas if --labels and --manifest provided.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sc_policy", default="prm_weighted",
                    choices=["majority", "prm_weighted", "prm_argmax", "consensus_or_prm"])
    ap.add_argument("--tr_policy", default="hint_majority",
                    choices=["hint_majority", "conclusion_text", "hybrid"])
    ap.add_argument("--min_steps", type=int, default=3)
    ap.add_argument("--min_strength", type=float, default=0.5)
    ap.add_argument("--max_original_frac", type=float, default=0.10)
    args = ap.parse_args()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tmp:
        sc_path = tmp.name

    # Step 1 — self-consistency (no-op if no candidates present)
    subprocess.run([
        sys.executable, "-m", "src.inference.self_consistency",
        "--input", args.input, "--output", sc_path,
        "--policy", args.sc_policy,
    ], check=True)

    # Step 2 — trace-rescue
    subprocess.run([
        sys.executable, "-m", "src.inference.trace_rescue",
        "--input", sc_path, "--output", args.output,
        "--policy", args.tr_policy,
        "--min_steps", str(args.min_steps),
        "--min_strength", str(args.min_strength),
        "--max_original_frac", str(args.max_original_frac),
    ], check=True)

    Path(sc_path).unlink(missing_ok=True)
    print(f"[stack] -> {args.output}")


if __name__ == "__main__":
    main()
