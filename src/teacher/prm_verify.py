"""Phase 1.3 — Verify the fine-tuned DDI-PRM checkpoint.

Loads the LoRA adapter from `--adapter` on top of `--base_model`, scores a
held-out eval set, and reports the gate metrics:

    - step-level token accuracy   (target ≥ 0.85)
    - step-level AUROC            (target ≥ 0.80)
    - step-level Brier            (lower is better; reported, not gated)

Also breaks down accuracy by family and tier so you can see if the PRM is
weak on a specific class. Emits a JSON report and a one-line PASS/FAIL.

Why this script (and not just the in-training eval_loss)
--------------------------------------------------------
The in-training eval_loss is on the same noisy auto-labels the model was
trained on, so a low loss alone doesn't prove the PRM agrees with humans.
This script:
  1. Computes step-level accuracy and AUROC, not just LM loss.
  2. Optionally compares against an unadapted (raw Med-PRM) baseline so you
     can quote the lift from DDI fine-tuning in the paper.

Usage on the cluster (single GPU is enough; ~25 min on 5k eval examples)
------------------------------------------------------------------------
    python -m src.teacher.prm_verify \\
        --adapter    $DDI_CKPT/ddi_prm_v1 \\
        --base_model dmis-lab/llama-3.1-medprm-reward-v1.0 \\
        --eval_file  $DDI_OUTPUTS/teacher/prm_eval.merged.jsonl \\
        --report     $DDI_OUTPUTS/teacher/ddi_prm_v1.verify.json \\
        --limit      5000

To also report the unadapted baseline (slower, runs the eval twice):
    --include_baseline
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    _GPU_OK = True
except ImportError as e:
    _GPU_OK = False
    _IMPORT_ERR = e


SYSTEM_PROMPT = (
    "You are an evaluator assessing the logicality and validity of each step "
    "of the following DDI reasoning trace.  For each reasoning step, output + "
    "if the step is logically valid and evidence-grounded; output - if the "
    "step contains an error (hallucinated IDs, flipped direction, off-family "
    "family_hint, silent abstention).  In addition, the question block "
    "contains the query pair and supporting evidence context."
)
SEP_TOKEN = " ки"


# ───────────────────── scorer (LoRA-aware) ─────────────────────
class PrmScorer:
    """Loads base + LoRA adapter and scores Med-PRM-formatted (q, sol) pairs."""

    def __init__(self, base_model: str, adapter_path: str | None,
                 device: str = "cuda:0", max_len: int = 3072):
        if not _GPU_OK:
            raise RuntimeError(f"GPU deps missing: {_IMPORT_ERR}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path or base_model, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "sdpa"

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl, device_map=device,
        )
        base.config.use_cache = False
        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path)
            print(f"[verify] loaded base + LoRA adapter from {adapter_path}",
                  flush=True)
        else:
            self.model = base
            print("[verify] loaded base only (no adapter — baseline mode)",
                  flush=True)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.plus_id = self.tokenizer(" +", add_special_tokens=False)["input_ids"][0]
        self.minus_id = self.tokenizer(" -", add_special_tokens=False)["input_ids"][0]
        self.max_len = max_len

    @torch.no_grad()
    def score(self, question: str, solution: str) -> list[float]:
        """Returns one P(+) per ` ки` separator in the solution.

        If the input was truncated past some separator, the corresponding
        positions are silently dropped — the caller can detect this by
        comparing len(probs) to the number of expected step labels.
        """
        text = (
            f"[System]\n{SYSTEM_PROMPT}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Solution]\n{solution}"
        )
        enc = self.tokenizer(text, return_offsets_mapping=True,
                              add_special_tokens=True, return_tensors="pt",
                              truncation=True, max_length=self.max_len)
        enc = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        offsets = self.tokenizer(text, return_offsets_mapping=True,
                                 add_special_tokens=True,
                                 truncation=True,
                                 max_length=self.max_len)["offset_mapping"]
        logits = self.model(**enc).logits[0]

        sep_positions: list[int] = []
        i = 0
        while True:
            j = text.find(SEP_TOKEN, i)
            if j < 0:
                break
            sep_positions.append(j)
            i = j + len(SEP_TOKEN)

        probs: list[float] = []
        for pos in sep_positions:
            ti = None
            for k, (s, e) in enumerate(offsets):
                if s <= pos < e:
                    ti = k
                    break
            if ti is None or ti >= logits.size(0):
                continue
            logit_pair = torch.stack([logits[ti][self.plus_id],
                                      logits[ti][self.minus_id]])
            p = torch.softmax(logit_pair, dim=0)[0].item()
            probs.append(p)
        return probs


# ───────────────────── metrics ─────────────────────
def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U-form AUROC. O(n log n). Pure python (no sklearn)."""
    if not scores or len(set(labels)) < 2:
        return float("nan")
    paired = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i
        while j + 1 < n and paired[j + 1][0] == paired[i][0]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            if paired[k][1] == 1:
                rank_sum_pos += rank
        i = j + 1
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def brier(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return float("nan")
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)


def accuracy(scores: list[float], labels: list[int],
             threshold: float = 0.5) -> float:
    if not scores:
        return float("nan")
    correct = sum(int((s >= threshold) == (y == 1)) for s, y in zip(scores, labels))
    return correct / len(scores)


# ───────────────────── eval loop ─────────────────────
def evaluate(scorer: PrmScorer, eval_path: Path, limit: int | None,
             progress_every: int = 100, seed: int = 13) -> dict:
    rows = []
    with eval_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]

    scores_all: list[float] = []
    labels_all: list[int] = []
    by_family: dict[str, tuple[list[float], list[int]]] = defaultdict(
        lambda: ([], []))
    by_tier: dict[str, tuple[list[float], list[int]]] = defaultdict(
        lambda: ([], []))

    n_done = 0
    n_drop = 0
    for r in rows:
        probs = scorer.score(r["question"], r["solution"])
        gold = r["step_labels"]
        n = min(len(probs), len(gold))
        if n == 0:
            n_drop += 1
            continue
        n_drop += max(0, len(gold) - len(probs))
        s = probs[:n]
        y = [int(bool(b)) for b in gold[:n]]
        scores_all.extend(s)
        labels_all.extend(y)
        fam = r.get("family") or "unknown"
        by_family[fam][0].extend(s)
        by_family[fam][1].extend(y)
        tier = "strict" if r.get("trace_strict_passed") else "loose"
        by_tier[tier][0].extend(s)
        by_tier[tier][1].extend(y)

        n_done += 1
        if n_done % progress_every == 0:
            running_acc = accuracy(scores_all, labels_all)
            print(f"[verify] {n_done}/{len(rows)}  running acc={running_acc:.3f}",
                  flush=True)

    return {
        "n_traces":       n_done,
        "n_steps":        len(scores_all),
        "n_steps_dropped_truncation": n_drop,
        "metrics": {
            "step_accuracy": accuracy(scores_all, labels_all),
            "step_auroc":    auroc(scores_all, labels_all),
            "step_brier":    brier(scores_all, labels_all),
        },
        "per_family": {
            fam: {
                "n":     len(lbls),
                "acc":   accuracy(scrs, lbls),
                "auroc": auroc(scrs, lbls),
            }
            for fam, (scrs, lbls) in by_family.items()
        },
        "per_tier": {
            tier: {
                "n":     len(lbls),
                "acc":   accuracy(scrs, lbls),
                "auroc": auroc(scrs, lbls),
            }
            for tier, (scrs, lbls) in by_tier.items()
        },
    }


def gate_check(metrics: dict, tok_acc_target: float, auroc_target: float) -> dict:
    acc = metrics["step_accuracy"]
    au = metrics["step_auroc"]
    return {
        "tok_acc_target": tok_acc_target,
        "tok_acc":        acc,
        "tok_acc_pass":   bool(acc >= tok_acc_target),
        "auroc_target":   auroc_target,
        "auroc":          au,
        "auroc_pass":     bool(au >= auroc_target),
        "overall_pass":   bool(acc >= tok_acc_target and au >= auroc_target),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True,
                    help="Path to the LoRA adapter dir saved by Trainer.")
    ap.add_argument("--base_model",
                    default="dmis-lab/llama-3.1-medprm-reward-v1.0")
    ap.add_argument("--eval_file", required=True,
                    help="prm_eval.merged.jsonl built by src/teacher/prm_data.py")
    ap.add_argument("--report", required=True,
                    help="Path to write the JSON report.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=5000,
                    help="Score the first N traces (after seeded shuffle). "
                         "5000 gives ~±0.005 CI on accuracy. Set 0 for full set.")
    ap.add_argument("--include_baseline", action="store_true",
                    help="Also score the unadapted base model. ~2x runtime.")
    ap.add_argument("--tok_acc_target", type=float, default=0.85)
    ap.add_argument("--auroc_target", type=float, default=0.80)
    ap.add_argument("--max_len", type=int, default=3072)
    args = ap.parse_args()

    eval_path = Path(args.eval_file)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    limit = args.limit if args.limit and args.limit > 0 else None

    out = {
        "adapter":     args.adapter,
        "base_model":  args.base_model,
        "eval_file":   str(eval_path),
        "limit":       limit,
    }

    # 1. Adapted PRM
    scorer = PrmScorer(args.base_model, args.adapter,
                       device=args.device, max_len=args.max_len)
    adapted = evaluate(scorer, eval_path, limit)
    out["adapted"] = adapted
    out["adapted"]["gate"] = gate_check(
        adapted["metrics"], args.tok_acc_target, args.auroc_target)
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. (optional) baseline PRM, no adapter
    if args.include_baseline:
        baseline_scorer = PrmScorer(args.base_model, None,
                                    device=args.device, max_len=args.max_len)
        baseline = evaluate(baseline_scorer, eval_path, limit)
        out["baseline"] = baseline
        out["lift"] = {
            "step_accuracy_delta": adapted["metrics"]["step_accuracy"]
                                   - baseline["metrics"]["step_accuracy"],
            "step_auroc_delta":    adapted["metrics"]["step_auroc"]
                                   - baseline["metrics"]["step_auroc"],
        }

    with report_path.open("w") as f:
        json.dump(out, f, indent=2)

    # ──────────────── pretty console summary ────────────────
    g = out["adapted"]["gate"]
    m = out["adapted"]["metrics"]
    print("")
    print("=" * 60)
    print(f"[verify] DDI-PRM gate check ({adapted['n_traces']:,} traces, "
          f"{adapted['n_steps']:,} steps)")
    print("=" * 60)
    print(f"  step accuracy:  {m['step_accuracy']:.4f}   "
          f"(target ≥ {args.tok_acc_target}) "
          f"{'PASS' if g['tok_acc_pass'] else 'FAIL'}")
    print(f"  step AUROC:     {m['step_auroc']:.4f}   "
          f"(target ≥ {args.auroc_target}) "
          f"{'PASS' if g['auroc_pass'] else 'FAIL'}")
    print(f"  step Brier:     {m['step_brier']:.4f}   (lower is better)")
    if "lift" in out:
        print(f"  lift over baseline:  Δacc={out['lift']['step_accuracy_delta']:+.4f}  "
              f"Δauroc={out['lift']['step_auroc_delta']:+.4f}")
    print("")
    print(f"  GATE: {'PASS — proceed to Phase 2.1 (real critic)' if g['overall_pass'] else 'FAIL — see per-family / per-tier in report'}")
    print(f"  report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
