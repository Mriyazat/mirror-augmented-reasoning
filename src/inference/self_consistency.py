"""Self-consistency aggregator over rerank candidates.

Reads a rerank4/8 prediction file and emits a new prediction file where the
final family is chosen by one of:

  majority      — strict plurality vote across candidates.
  prm_weighted  — vote weighted by candidate's prm_final.
  prm_argmax    — the candidate with highest prm_final (== existing default).
  consensus_or_prm — if any family has >= consensus_thresh share use it;
                     else fall back to prm_argmax.

The picked candidate's trace/subtype/direction are kept for downstream use.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def pick(candidates: list[dict], policy: str, consensus_thresh: float = 0.5):
    if not candidates:
        return None, None
    valid = [c for c in candidates if c.get("parse_ok") and c.get("family")]
    if not valid:
        return candidates[0], "fallback_unparsed"

    if policy == "prm_argmax":
        best = max(valid, key=lambda c: float(c.get("prm_final") or 0.0))
        return best, "prm_argmax"

    if policy == "majority":
        c = Counter(c["family"] for c in valid)
        winner, _ = c.most_common(1)[0]
        chosen = max((cc for cc in valid if cc["family"] == winner),
                     key=lambda x: float(x.get("prm_final") or 0.0))
        return chosen, "majority"

    if policy == "prm_weighted":
        scores = defaultdict(float)
        for c in valid:
            scores[c["family"]] += max(float(c.get("prm_final") or 0.0), 0.0)
        if not scores or all(v == 0 for v in scores.values()):
            return pick(candidates, "prm_argmax")
        winner = max(scores, key=scores.get)
        chosen = max((cc for cc in valid if cc["family"] == winner),
                     key=lambda x: float(x.get("prm_final") or 0.0))
        return chosen, "prm_weighted"

    if policy == "consensus_or_prm":
        c = Counter(c["family"] for c in valid)
        winner, n_winner = c.most_common(1)[0]
        if n_winner / len(valid) >= consensus_thresh:
            chosen = max((cc for cc in valid if cc["family"] == winner),
                         key=lambda x: float(x.get("prm_final") or 0.0))
            return chosen, "consensus"
        return pick(candidates, "prm_argmax")

    raise ValueError(f"unknown policy {policy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", choices=["majority", "prm_weighted", "prm_argmax", "consensus_or_prm"], default="prm_weighted")
    ap.add_argument("--consensus_thresh", type=float, default=0.5)
    args = ap.parse_args()

    n_rec = n_changed = 0
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, op.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rec += 1
            rerank = r.get("rerank") or {}
            cands = rerank.get("candidates") or []
            if not cands:
                fout.write(json.dumps(r) + "\n")
                continue
            chosen, rule = pick(cands, args.policy, args.consensus_thresh)
            if not chosen:
                fout.write(json.dumps(r) + "\n")
                continue
            old_fam = (r.get("final_prediction") or {}).get("family")
            new_fam = chosen.get("family")
            if new_fam and new_fam != old_fam:
                n_changed += 1
            fp = r.setdefault("final_prediction", {})
            if new_fam:
                fp["family"] = new_fam
            if chosen.get("subtype"):
                fp["subtype"] = chosen.get("subtype")
            if chosen.get("direction_tag"):
                fp["direction_tag"] = chosen.get("direction_tag")
            r["self_consistency"] = {"rule": rule, "old": old_fam, "new": new_fam}
            fout.write(json.dumps(r) + "\n")

    print(f"[sc] {n_changed}/{n_rec} families changed by {args.policy}")
    print(f"[sc] wrote {op}")


if __name__ == "__main__":
    main()
