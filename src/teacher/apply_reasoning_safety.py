"""Phase 2.3 — Apply reasoning-safety pass to a consensus corpus.

Inputs:
  --teacher_clean   the consensus JSONL produced by `merge_consensus.py`
  --audit           the per-trace audit JSONL produced by
                    `audit_teacher_clean.py` (`b7_audit_per_trace.jsonl`)

Outputs:
  <output> (JSONL)              the cleaned, downweighted SFT corpus
  <output>.report.md            human-readable filter report

Two-step processing:

  1. **Drop** any record with a hard-quality flag.  We only drop on flags
     that indicate the trace is meaningfully broken (not just stylistically
     thin):
        schema_fail
        direction_inconsistent
        abstention_high_confidence
        no_drug_name_in_summary
        ctx_build_fail:*
     This typically removes <0.1% of the corpus.

  2. **Rescale** `sample_weight *= quality_score` for everything that
     survives.  Effects:
        - Records with quality_score=1.0 keep their full consensus weight.
        - Records with weak_mechanism_skeleton (qs ~ 0.85) get a 15%
          haircut.
        - Records with multiple soft flags (qs ~ 0.7) get a 30% haircut.
     Final weight is clamped to [0, original_weight] so we never *boost*
     a record above its consensus-merge weight.

Why both?  Hard bugs (e.g. direction_inconsistent) shouldn't reach the
student at all, no matter how much consensus they had.  Soft thin traces
should still teach diversity, just at lower influence on the loss.

Usage
-----
    python -m src.teacher.apply_reasoning_safety \\
        --teacher_clean $DDI_OUTPUTS/teacher/teacher_clean.consensus.jsonl \\
        --audit         outputs/audit/b7_audit_per_trace.jsonl \\
        --output        $DDI_OUTPUTS/teacher/teacher_clean.consensus.reasoning_safe.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


HARD_DROP_FLAGS = {
    "schema_fail",
    "direction_inconsistent",
    "abstention_high_confidence",
    "no_drug_name_in_summary",
}


def _has_hard_flag(audit: dict) -> tuple[bool, list[str]]:
    flags = audit.get("flags", []) or []
    triggered: list[str] = []
    for f in flags:
        # We accept either exact match or "<flag>=value" prefix-match
        # (e.g., the audit emits "abstention_high_confidence=0.78").
        head = f.split("=", 1)[0].split(":", 1)[0]
        if f in HARD_DROP_FLAGS or head in HARD_DROP_FLAGS:
            triggered.append(f)
        if head.startswith("ctx_build_fail"):
            triggered.append(f)
    return bool(triggered), triggered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher_clean", required=True,
                   help="Input consensus corpus (JSONL).")
    p.add_argument("--audit", required=True,
                   help="b7_audit_per_trace.jsonl from audit_teacher_clean.")
    p.add_argument("--output", required=True,
                   help="Output reasoning_safe corpus (JSONL).")
    p.add_argument("--qs_floor", type=float, default=0.30,
                   help="Drop records whose quality_score is below this. "
                        "Default 0.30 — well below the audit's suspect "
                        "threshold of 0.55, so this is a safety floor only "
                        "(corpus is already filtered through QC + tier).")
    p.add_argument("--no_rescale", action="store_true",
                   help="Skip the sample_weight *= quality_score rescaling. "
                        "Use only the hard-flag drop step.")
    args = p.parse_args()

    audit_lookup: dict[str, dict] = {}
    with open(args.audit) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            audit_lookup[a["pair_id"]] = a
    print(f"[reasoning_safety] loaded {len(audit_lookup):,} audit records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_drop_hard = 0
    n_drop_floor = 0
    n_drop_no_audit = 0
    n_rescaled = 0
    drop_flag_ct: Counter = Counter()
    qs_buckets: Counter = Counter()
    weight_before = 0.0
    weight_after = 0.0

    with open(args.teacher_clean) as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            pid = rec.get("pair_id")
            audit = audit_lookup.get(pid)
            if audit is None:
                n_drop_no_audit += 1
                continue

            qs = float(audit.get("quality_score") or 0.0)
            qs_buckets[round(qs * 10) / 10] += 1
            if qs < args.qs_floor:
                n_drop_floor += 1
                continue

            had_hard, triggered = _has_hard_flag(audit)
            if had_hard:
                n_drop_hard += 1
                for f in triggered:
                    drop_flag_ct[f.split("=", 1)[0].split(":", 1)[0]] += 1
                continue

            base_w = float(rec.get("sample_weight") or 0.0)
            weight_before += base_w
            if not args.no_rescale:
                new_w = max(0.0, min(base_w * qs, base_w))
                if abs(new_w - base_w) > 1e-9:
                    n_rescaled += 1
                rec["sample_weight"] = round(new_w, 4)
                rec["quality_score"] = qs
                rec["audit_flags"] = audit.get("flags", []) or []
                weight_after += new_w
            else:
                rec["quality_score"] = qs
                rec["audit_flags"] = audit.get("flags", []) or []
                weight_after += base_w

            fout.write(json.dumps(rec) + "\n")
            n_out += 1

    # ---------- report ----------
    report = [
        "# Phase 2.3 — Reasoning-safety pass report",
        "",
        f"- input corpus:  `{args.teacher_clean}`",
        f"- audit file:    `{args.audit}`",
        f"- output corpus: `{args.output}`",
        "",
        f"- input records:           **{n_in:,}**",
        f"- emitted records:         **{n_out:,}**  "
        f"({100 * n_out / max(1, n_in):.2f}%)",
        f"- dropped (hard-flag):     {n_drop_hard:,}",
        f"- dropped (qs < {args.qs_floor:.2f}): {n_drop_floor:,}",
        f"- dropped (no audit row):  {n_drop_no_audit:,}",
        "",
        f"- weight rescaled:         {n_rescaled:,} records",
        f"- total weight before:     {weight_before:.1f}",
        f"- total weight after:      {weight_after:.1f}  "
        f"({100 * weight_after / max(1e-9, weight_before):.2f}% of input)",
        "",
        "## Hard-flag drops by reason",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    for flag, c in drop_flag_ct.most_common():
        report.append(f"| `{flag}` | {c:,} |")
    if not drop_flag_ct:
        report.append("| (none) | 0 |")

    report += [
        "",
        "## Quality-score histogram (input)",
        "",
        "| qs bucket | count |",
        "|---|---:|",
    ]
    for k in sorted(qs_buckets):
        report.append(f"| {k:.1f} | {qs_buckets[k]:,} |")

    report_path = Path(args.output + ".report.md")
    report_path.write_text("\n".join(report) + "\n")
    print(f"[reasoning_safety] wrote {args.output}")
    print(f"[reasoning_safety] wrote {report_path}")
    print(f"[reasoning_safety] in={n_in:,} out={n_out:,} "
          f"hard_drop={n_drop_hard} floor_drop={n_drop_floor} "
          f"weight_kept={100 * weight_after / max(1e-9, weight_before):.1f}%")


if __name__ == "__main__":
    main()
