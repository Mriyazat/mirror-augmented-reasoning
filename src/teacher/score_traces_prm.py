"""B7 -- PRM scoring for teacher_clean traces.

Takes the merged teacher corpus (`teacher_clean.jsonl`, one record per
winner trace) and scores each trace through DDI-PRM, emitting per-step
and trace-level `+` probabilities.

Why this script?
----------------
Several downstream consumers need step-level PRM scores:

  - **SLFS metric** (`src/metrics/slfs.py`): aggregates step PRM
    scores into a trace-level faithfulness number.  Needs every
    published trace to have `steps[i].prm_score`.
  - **Mirror-DPO preference builder**
    (`src/teacher/build_preference_pairs.py`): takes `prm_chosen`
    from here to weight preference pairs.
  - **Ablation: "-- PRM weighting" row**: we need the PRM corpus to
    exist before the SFT/DPO ablations can be run.

Output format (`prm_scores.jsonl`)
----------------------------------
    {
      "pair_id":     str,
      "prm_mean":    float,    # mean over steps (incl. final "ORM" step)
      "prm_min":     float,    # min_plus  -- same semantics as critic.py
      "prm_final":   float,    # probability on the final conclusion step
      "n_steps":     int,
      "steps": [
          {"step_id": 1, "role": "pathway", "prm_score": 0.87},
          {"step_id": 2, "role": "pk_flag", "prm_score": 0.92},
          ...
          {"step_id": 99, "role": "final_answer", "prm_score": 0.81}
      ]
    }

Med-PRM formatting
------------------
We mirror `src/teacher/prm_data.trace_to_medprm_example` but operate
on the ASSISTANT JSON in `teacher_clean.jsonl` (which has already been
QC'd and merged).  The "question" is reconstructed from the user
message (which contains the pair context), and the "solution" is
the step-by-step rationale ending with ` ки` separators.

CLI
---
    python -m src.teacher.score_traces_prm \
        --teacher_clean outputs/teacher/teacher_clean.jsonl \
        --prm_model     dmis-lab/llama-3.1-medprm-reward-v1.0 \
        --output        outputs/teacher/prm_scores.jsonl \
        --batch_every   200 \
        --device        cuda:0

To smoke-test without GPU / model download:
    python -m src.teacher.score_traces_prm \
        --teacher_clean <some.jsonl> \
        --output         <out.jsonl> \
        --dry_run
(emits QC-fallback pseudo-scores, same shape as the real output.)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from src.teacher.schema import extract_json_block, load_rubric

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRM_MODEL = "dmis-lab/llama-3.1-medprm-reward-v1.0"


# ======================================================================
# Med-PRM formatting -- reused across teacher_clean -> PRM scoring.
# ======================================================================
def _build_medprm_inputs(record: dict, sep: str) -> tuple[str, str, list[dict]] | None:
    """From a teacher_clean record, produce (question, solution, step_meta).

    step_meta is a list aligned with the `+`/`-` separators in the
    solution, each {step_id, role}.  The PRM will score this sequence
    and return a list of probabilities; we re-join on step_meta.
    """
    msgs = record.get("messages") or []
    user_msg = next((m for m in msgs if m.get("role") == "user"), None)
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if asst is None:
        return None
    try:
        parsed = extract_json_block(asst.get("content") or "")
    except Exception:
        return None
    if not isinstance(parsed, dict) or "final_answer" not in parsed:
        return None

    question = (user_msg or {}).get("content", "").strip() or f"pair_id={record.get('pair_id','?')}"

    sol_lines: list[str] = []
    step_meta: list[dict] = []
    for s in parsed.get("steps", []) or []:
        sid = s.get("step_id")
        role = s.get("role", "unknown")
        claim = (s.get("claim") or "").strip()
        ev = (s.get("evidence_ids") or [])
        dt = s.get("direction_tag", "n/a")
        ev_str = f" [evidence: {', '.join(ev)}]" if ev else ""
        dt_str = f" ({dt})" if dt and dt != "n/a" else ""
        sol_lines.append(f"Step {sid}: {claim}{ev_str}{dt_str}{sep}")
        step_meta.append({"step_id": sid, "role": role})

    ans = parsed["final_answer"]
    sol_lines.append(
        f"Final: family={ans.get('family','')}, subtype={ans.get('subtype','')}, "
        f"direction={ans.get('direction_tag','')}, "
        f"polarity={ans.get('polarity','')}, "
        f"abstain={ans.get('abstain', False)}{sep}"
    )
    step_meta.append({"step_id": "final", "role": "final_answer"})

    return question, "\n".join(sol_lines), step_meta


# ======================================================================
# Dry-run fallback (no PRM loaded)
# ======================================================================
def _qc_fallback_probs(record: dict, step_meta: list[dict]) -> list[float]:
    """Synthesize pseudo-scores using record-level tier + any available
    qc hints.  Used by --dry_run and also when the PRM is unavailable
    at runtime (so the pipeline still emits a file of the right shape).

    Mapping (conservative, matches critic.qc_fallback_score ranges):
        tier=full_correct   -> [0.92, ..., 0.93]
        tier=family_correct -> [0.78, ..., 0.85]
        tier=abstention     -> [0.70, ..., 0.80]
        tier=near_miss      -> [0.45, ..., 0.55]
        unknown             -> [0.50, ..., 0.60]
    Small per-step jitter keeps variance non-zero so downstream
    aggregators (mean/min/min_plus) don't collapse to a single number.
    """
    tier = record.get("tier", "unknown")
    base = {
        "full_correct":   (0.88, 0.94),
        "family_correct": (0.76, 0.84),
        "abstention":     (0.70, 0.80),
        "near_miss":      (0.45, 0.55),
    }.get(tier, (0.50, 0.60))

    import random
    seed = abs(hash(record.get("pair_id", "x"))) % (2**31)
    rng = random.Random(seed)
    lo, hi = base
    return [round(rng.uniform(lo, hi), 3) for _ in step_meta]


# ======================================================================
# Main scoring loop
# ======================================================================
def score_file(teacher_clean: Path, out_path: Path,
               prm_model_path: str | None,
               dry_run: bool, device: str,
               limit: int | None = None,
               progress_every: int = 100) -> dict:
    rubric = load_rubric()
    sep = rubric["separator_token"]

    scorer = None
    if not dry_run:
        try:
            from src.teacher.critic import PRMScorer
            scorer = PRMScorer(prm_model_path or DEFAULT_PRM_MODEL, device=device)
            print(f"[prm] loaded PRM from {prm_model_path or DEFAULT_PRM_MODEL} on {device}")
        except Exception as e:
            print(f"[prm] WARN: failed to load PRM ({e}); falling back to dry-run scores.")
            scorer = None
            dry_run = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_fail = 0

    with open(teacher_clean) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if limit is not None and n_out >= limit:
                break
            record = json.loads(line)

            packed = _build_medprm_inputs(record, sep)
            if packed is None:
                n_fail += 1
                continue
            question, solution, step_meta = packed

            if dry_run or scorer is None:
                probs = _qc_fallback_probs(record, step_meta)
            else:
                scored = scorer.score(question, solution)
                probs = list(scored.get("step_probs") or [])
                # PRMScorer may return fewer probs than step_meta if
                # truncation happened; pad with final_plus.
                if len(probs) < len(step_meta):
                    tail = scored.get("final_plus", 0.0)
                    probs = probs + [tail] * (len(step_meta) - len(probs))
                elif len(probs) > len(step_meta):
                    probs = probs[: len(step_meta)]

            steps_out = [
                {"step_id": m["step_id"], "role": m["role"],
                 "prm_score": float(probs[i])}
                for i, m in enumerate(step_meta)
            ]
            prm_mean = sum(probs) / len(probs) if probs else 0.0
            prm_min = min(probs) if probs else 0.0
            prm_final = probs[-1] if probs else 0.0

            fout.write(json.dumps({
                "pair_id":   record.get("pair_id"),
                "prm_mean":  prm_mean,
                "prm_min":   prm_min,
                "prm_final": prm_final,
                "n_steps":   len(steps_out),
                "steps":     steps_out,
                "tier":      record.get("tier"),
                "dry_run":   dry_run,
            }) + "\n")
            n_out += 1

            if n_out % progress_every == 0:
                print(f"[prm] scored {n_out:,} traces...")

    print(f"[prm] read {n_in:,}; scored {n_out:,}; failed {n_fail:,}.  -> {out_path}")
    return {"n_in": n_in, "n_out": n_out, "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--prm_model",     default=DEFAULT_PRM_MODEL)
    ap.add_argument("--device",        default="cuda:0")
    ap.add_argument("--dry_run", action="store_true",
                    help="Emit QC-fallback pseudo-scores instead of calling the PRM.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N traces.  For debugging.")
    ap.add_argument("--progress_every", type=int, default=100)
    args = ap.parse_args()

    score_file(
        teacher_clean=Path(args.teacher_clean),
        out_path=Path(args.output),
        prm_model_path=args.prm_model,
        dry_run=args.dry_run,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
