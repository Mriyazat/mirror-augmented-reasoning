"""Re-rank a critic_*.jsonl file post-hoc using a different aggregation.

Why this exists
===============

`src/teacher/critic.py` runs the PRM on every QC-survived candidate, picks a
winner per pair, and writes one record per pair to a `critic_*.jsonl` file.
Each record stores the full per-candidate score vector under
`all_candidate_scores`, so we can change the aggregation/ranker WITHOUT
re-running the PRM.

Empirically the PRM was trained step-level with many strict-tier examples,
so its per-step P(+) saturates near 0 for most intermediate steps.  This
makes step-level aggregations (`min_plus`, `geomean_plus`) collapse to the
noise floor — the sort then picks winners essentially at random.

The PRM's FINAL step, in contrast, was supervised on `critical_passed`
(whole-trace correctness).  It is effectively an ORM-style head with a
healthy distribution (mean ≈ 0.48, p10 ≈ 0.0, p90 ≈ 0.95), and it
discriminates traces well.  Diagnostic on 6,000 pairs showed switching
the primary key from geomean_plus to final_plus changes the winner in
~73% of pairs, with 27% gaining >0.1 in final_plus.  That's a lot of
salvageable signal.

Usage
=====

  python -m src.teacher.critic_rerank \
      --critic $DDI_OUTPUTS/teacher/critic_<TEACHER>.jsonl \
      --qc     $DDI_OUTPUTS/teacher/qc_subset25k_<TEACHER>.jsonl \
      --out    $DDI_OUTPUTS/teacher/critic_<TEACHER>.reranked.jsonl

Or sweep the three teachers in one shot — see the launcher block at the
bottom of `docs/V4_RECOVERY_PLAN.md` (Phase 2.1.5).

Note
====
This script only changes which candidate is marked as the winner of each
pair; it does NOT re-run the PRM and does NOT touch step_probs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter


SECONDARY_KEYS = ("geomean_plus", "mean_plus", "min_plus")

# Allowed values for --primary.
PRIMARY_CHOICES = (
    "final_plus",
    "geomean_plus",
    "mean_plus",
    "min_plus",
    "qc_then_final",   # qc_passed > qc_critical_passed > final_plus  (tiered)
)


def load_qc_lookup(qc_path: str) -> dict:
    """Build (pair_id, candidate_id) -> full QC rec lookup.

    The QC file holds ~600k records (24 candidates × 25k pairs); ~200 MB
    in memory after deserialization.  Streaming once is fine.
    """
    print(f"[rerank] indexing QC by (pair_id, candidate_id) from {qc_path}",
          flush=True)
    lookup: dict[tuple[str, str], dict] = {}
    with open(qc_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("pair_id")
            cid = r.get("candidate_id")
            if pid is None or cid is None:
                continue
            lookup[(pid, cid)] = r
    print(f"[rerank]   loaded {len(lookup):,} QC records", flush=True)
    return lookup


def _key(primary: str):
    """Build a sort key tuple.

    For the simple modes (final_plus, geomean_plus, mean_plus, min_plus) the
    key is (primary, then the other 3 score keys in order).

    For 'qc_then_final' the key is a hard tiered ordering:
        ( qc_passed,            # strict pass first
          qc_critical_passed,   # then critical pass
          final_plus,           # PRM ORM-style head
          geomean_plus,         # PRM step aggregate
          mean_plus,
          min_plus )
    All bool/float, all sorted descending.  This guarantees we never pick a
    QC-failing candidate when a QC-passing one is available in the same pair.
    """
    if primary == "qc_then_final":
        def keyfn_tiered(c: dict):
            return (
                int(bool(c.get("qc_passed"))),
                int(bool(c.get("qc_critical_passed"))),
                c["final_plus"],
                c["geomean_plus"],
                c["mean_plus"],
                c["min_plus"],
            )
        return keyfn_tiered

    secondary = [k for k in SECONDARY_KEYS + ("final_plus",) if k != primary]
    secondary = secondary[:3]

    def keyfn(c: dict):
        return tuple([c[primary]] + [c[k] for k in secondary])

    return keyfn


def rerank(
    critic_path: str,
    qc_lookup: dict,
    out_path: str,
    primary: str,
) -> dict:
    sort_key = _key(primary)
    n_total = 0
    n_changed = 0
    n_missing = 0
    n_no_cands = 0
    delta_sum = 0.0
    delta_pos = 0
    final_old = []
    final_new = []
    qc_pass_strict_old = 0
    qc_pass_strict_new = 0
    qc_pass_critical_old = 0
    qc_pass_critical_new = 0

    with open(critic_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            n_total += 1
            cands = rec.get("all_candidate_scores")
            if not cands:
                n_no_cands += 1
                fout.write(json.dumps(rec) + "\n")
                continue

            cands_sorted = sorted(cands, key=sort_key, reverse=True)
            new_score = cands_sorted[0]
            new_id = new_score["candidate_id"]

            old_winner = rec.get("winner_candidate") or {}
            old_id = old_winner.get("candidate_id")
            old_final = float(rec.get("score", {}).get("final_plus", 0.0))
            new_final = float(new_score["final_plus"])

            final_old.append(old_final)
            final_new.append(new_final)

            qc_pass_strict_old += int(bool(old_winner.get("passed")))
            qc_pass_critical_old += int(bool(old_winner.get("critical_passed")))

            if old_id != new_id:
                n_changed += 1
                pair_id = rec.get("pair_id")
                new_full = qc_lookup.get((pair_id, new_id))
                if new_full is None:
                    n_missing += 1
                    qc_pass_strict_new += int(bool(old_winner.get("passed")))
                    qc_pass_critical_new += int(
                        bool(old_winner.get("critical_passed")))
                    fout.write(json.dumps(rec) + "\n")
                    continue
                rec["winner_candidate"] = new_full
                rec["score"] = {
                    "min_plus":     new_score["min_plus"],
                    "mean_plus":    new_score["mean_plus"],
                    "geomean_plus": new_score["geomean_plus"],
                    "final_plus":   new_score["final_plus"],
                    "step_probs":   new_score["step_probs"],
                }
                qc_pass_strict_new += int(bool(new_full.get("passed")))
                qc_pass_critical_new += int(bool(new_full.get("critical_passed")))
                d = new_final - old_final
                delta_sum += d
                if d > 0:
                    delta_pos += 1
            else:
                qc_pass_strict_new += int(bool(old_winner.get("passed")))
                qc_pass_critical_new += int(
                    bool(old_winner.get("critical_passed")))

            fout.write(json.dumps(rec) + "\n")

    def _safe_mean(xs):
        return sum(xs) / max(1, len(xs))

    stats = {
        "input_pairs": n_total,
        "winners_changed": n_changed,
        "winners_changed_pct": round(100 * n_changed / max(1, n_total), 2),
        "missing_qc_lookup": n_missing,
        "pairs_with_no_candidates": n_no_cands,
        "primary_key": primary,
        "mean_final_plus_before": round(_safe_mean(final_old), 4),
        "mean_final_plus_after":  round(_safe_mean(final_new), 4),
        "mean_final_plus_gain":   round(
            _safe_mean(final_new) - _safe_mean(final_old), 4),
        "n_winners_with_higher_final": delta_pos,
        "qc_strict_pass_rate_before": round(
            100 * qc_pass_strict_old / max(1, n_total), 2),
        "qc_strict_pass_rate_after": round(
            100 * qc_pass_strict_new / max(1, n_total), 2),
        "qc_critical_pass_rate_before": round(
            100 * qc_pass_critical_old / max(1, n_total), 2),
        "qc_critical_pass_rate_after": round(
            100 * qc_pass_critical_new / max(1, n_total), 2),
    }
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--critic", required=True,
        help="Path to a critic_<teacher>.jsonl produced by src/teacher/critic.py.")
    ap.add_argument(
        "--qc", required=True,
        help="The QC file the critic was run against.  Needed to look up the "
             "full record for the new winner.")
    ap.add_argument(
        "--out", required=True,
        help="Output critic_<teacher>.reranked.jsonl path.")
    ap.add_argument(
        "--primary", default="qc_then_final",
        choices=list(PRIMARY_CHOICES),
        help="Primary sort key.  Default 'qc_then_final' is a tiered ranker: "
             "qc_passed > qc_critical_passed > final_plus.  This guarantees "
             "we never demote a QC-passing candidate in favor of a QC-failing "
             "one with a marginally higher PRM score.  Use 'final_plus' "
             "(PRM ORM-style head) for pure PRM ranking, or 'geomean_plus' "
             "to reproduce the in-loop critic.py ordering.")
    args = ap.parse_args()

    if not os.path.exists(args.critic):
        sys.exit(f"[rerank] critic file not found: {args.critic}")
    if not os.path.exists(args.qc):
        sys.exit(f"[rerank] qc file not found: {args.qc}")

    qc_lookup = load_qc_lookup(args.qc)
    stats = rerank(args.critic, qc_lookup, args.out, args.primary)

    print("[rerank] " + "─" * 60)
    print(f"[rerank] in:  {args.critic}")
    print(f"[rerank] out: {args.out}")
    print(f"[rerank] primary key: {stats['primary_key']}")
    print(f"[rerank] pairs: {stats['input_pairs']:,}")
    print(f"[rerank] winners changed: {stats['winners_changed']:,} "
          f"({stats['winners_changed_pct']:.1f}%)")
    if stats["missing_qc_lookup"]:
        print(f"[rerank] WARN missing QC for {stats['missing_qc_lookup']:,} "
              f"pairs (kept old winner for these)")
    print(f"[rerank] mean(final_plus): "
          f"{stats['mean_final_plus_before']:.4f} → "
          f"{stats['mean_final_plus_after']:.4f} "
          f"(Δ +{stats['mean_final_plus_gain']:.4f})")
    print(f"[rerank] QC strict pass rate of winner: "
          f"{stats['qc_strict_pass_rate_before']:.2f}% → "
          f"{stats['qc_strict_pass_rate_after']:.2f}%")
    print(f"[rerank] QC critical pass rate of winner: "
          f"{stats['qc_critical_pass_rate_before']:.2f}% → "
          f"{stats['qc_critical_pass_rate_after']:.2f}%")
    print("[rerank] " + "─" * 60)

    report_path = args.out + ".rerank_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[rerank] full report → {report_path}", flush=True)


if __name__ == "__main__":
    main()
