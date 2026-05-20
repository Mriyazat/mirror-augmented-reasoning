"""Re-aggregate the N candidates from a `predict_with_rerank` output
into multiple alternative final predictions.  Pure CPU: re-reads the
per-candidate trace+PRM bundle already in the file and writes new
JSONLs that are directly evaluable with `run_full_eval.py`.

Variants
--------
Baseline:
  greedy                  -> chosen_idx = 0  (simulates N=1 with temp=0)
  prm_argmax_n{1,2,4,8}   -> first N candidates only, then PRM argmax
  vote_majority           -> majority family across N; PRM-argmax inside
                             the winning family
  vote_then_prm_tiebreak  -> majority if a strict majority exists, else
                             PRM-argmax over the full N

Novel (for the inference-time scaling story in S6):
  vote_prm_weighted       -> each candidate votes for its family with
                             weight = PRM score; subtype/direction from
                             top-PRM in the winning family.
  prm_vote_consensus      -> commit only if top-PRM candidate's family
                             equals the majority-vote family; else mark
                             abstain=True.  Zero calibration needed.
  vote_margin_score       -> same family decision as vote_majority, but
                             writes (top_votes - second_votes) / N into
                             final_prediction.confidence so the conformal
                             layer can use it INSTEAD of PRM as the
                             abstention signal (cheaper at inference).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


def _load_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _candidate_score(cand: dict, metric: str) -> float:
    field = {
        "geomean_plus": "prm_geomean",
        "mean_plus":    "prm_mean",
        "min_plus":     "prm_min",
        "final_plus":   "prm_final",
    }.get(metric, "prm_geomean")
    v = cand.get(field)
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def _make_record(base: dict, picked_cand: dict, idx: int,
                 family_override: str | None,
                 metric: str, source_tag: str,
                 candidates_subset: list[dict] | None = None) -> dict:
    out = deepcopy(base)
    fp = out.setdefault("final_prediction", {})
    pf = picked_cand or {}
    fam = family_override if family_override is not None else pf.get("family")
    subtype = pf.get("subtype")
    direction = pf.get("direction_tag")
    if family_override is not None and family_override != pf.get("family"):
        subtype = None
        direction = "n/a"
    fp["family"] = fam
    fp["subtype"] = subtype
    fp["direction_tag"] = direction
    fp["abstain"] = False
    fp["aggregator"] = source_tag
    rer = out.setdefault("rerank", {})
    rer["aggregator"] = source_tag
    rer["chosen_idx"] = int(idx) if idx is not None else -1
    rer["chosen_prm"] = _candidate_score(pf, metric) if pf else 0.0
    rer["metric"] = metric
    if candidates_subset is not None:
        rer["n_candidates_used"] = len(candidates_subset)
    return out


def _greedy(base, cands, metric):
    if not cands:
        return _make_record(base, {}, -1, None, metric, "greedy")
    return _make_record(base, cands[0], 0, None, metric, "greedy")


def _prm_argmax(base, cands, n, metric):
    sub = cands[:n] if n > 0 else cands
    if not sub:
        return _make_record(base, {}, -1, None, metric, f"prm_argmax_n{n}")
    scored = [(1 if c.get("parse_ok") else 0,
               _candidate_score(c, metric),
               -i, i, c) for i, c in enumerate(sub)]
    scored.sort(reverse=True)
    _, _, _, idx, c = scored[0]
    return _make_record(base, c, idx, None, metric, f"prm_argmax_n{n}",
                        candidates_subset=sub)


def _vote_majority(base, cands, metric):
    if not cands:
        return _make_record(base, {}, -1, None, metric, "vote_majority")
    fams = [c.get("family") for c in cands if c.get("parse_ok") and c.get("family")]
    if not fams:
        return _prm_argmax(base, cands, len(cands), metric)
    counter = Counter(fams)
    top_fam, _ = counter.most_common(1)[0]
    fam_cands = [c for c in cands if c.get("family") == top_fam and c.get("parse_ok")]
    fam_cands.sort(key=lambda c: _candidate_score(c, metric), reverse=True)
    picked = fam_cands[0] if fam_cands else cands[0]
    picked_idx = cands.index(picked)
    return _make_record(base, picked, picked_idx, top_fam, metric,
                        "vote_majority", candidates_subset=cands)


def _vote_then_prm_tiebreak(base, cands, metric):
    if not cands:
        return _make_record(base, {}, -1, None, metric, "vote_then_prm_tiebreak")
    fams = [c.get("family") for c in cands if c.get("parse_ok") and c.get("family")]
    if not fams:
        return _prm_argmax(base, cands, len(cands), metric)
    counter = Counter(fams)
    _, top_n = counter.most_common(1)[0]
    threshold = len(cands) // 2  # strict majority
    if top_n > threshold:
        out = _vote_majority(base, cands, metric)
        out["final_prediction"]["aggregator"] = "vote_then_prm_tiebreak"
        return out
    return _prm_argmax(base, cands, len(cands), metric)


# ----------------------------------------------------------------------
# Novel aggregators (S6 of the paper)
# ----------------------------------------------------------------------
def _vote_prm_weighted(base, cands, metric):
    """Each candidate votes with weight = PRM score."""
    if not cands:
        return _make_record(base, {}, -1, None, metric, "vote_prm_weighted")
    weights: dict[str, float] = {}
    for c in cands:
        if not c.get("parse_ok"):
            continue
        f = c.get("family")
        if not f:
            continue
        weights[f] = weights.get(f, 0.0) + _candidate_score(c, metric)
    if not weights:
        return _prm_argmax(base, cands, len(cands), metric)
    top_fam = max(weights, key=weights.get)
    fam_cands = [c for c in cands if c.get("family") == top_fam and c.get("parse_ok")]
    fam_cands.sort(key=lambda c: _candidate_score(c, metric), reverse=True)
    picked = fam_cands[0] if fam_cands else cands[0]
    picked_idx = cands.index(picked)
    out = _make_record(base, picked, picked_idx, top_fam, metric,
                       "vote_prm_weighted", candidates_subset=cands)
    out["final_prediction"]["vote_weights"] = {k: float(v) for k, v in weights.items()}
    return out


def _prm_vote_consensus(base, cands, metric):
    """Commit only if top-PRM and majority-vote families agree."""
    if not cands:
        return _make_record(base, {}, -1, None, metric, "prm_vote_consensus")
    fams = [c.get("family") for c in cands if c.get("parse_ok") and c.get("family")]
    if not fams:
        out = _make_record(base, cands[0], 0, None, metric, "prm_vote_consensus")
        out["final_prediction"]["abstain"] = True
        return out
    counter = Counter(fams)
    top_vote_fam, top_vote_n = counter.most_common(1)[0]
    scored = sorted(
        [(i, c) for i, c in enumerate(cands) if c.get("parse_ok")],
        key=lambda ic: _candidate_score(ic[1], metric),
        reverse=True,
    )
    if not scored:
        out = _make_record(base, cands[0], 0, None, metric, "prm_vote_consensus")
        out["final_prediction"]["abstain"] = True
        return out
    top_prm_idx, top_prm_cand = scored[0]
    consensus = (top_prm_cand.get("family") == top_vote_fam)
    out = _make_record(base, top_prm_cand, top_prm_idx,
                       top_vote_fam if consensus else None,
                       metric, "prm_vote_consensus",
                       candidates_subset=cands)
    fp = out["final_prediction"]
    fp["abstain"] = (not consensus)
    fp["vote_margin"] = float(top_vote_n) / max(len(cands), 1)
    fp["prm_top_fam"] = top_prm_cand.get("family")
    fp["vote_top_fam"] = top_vote_fam
    fp["consensus"] = bool(consensus)
    return out


def _vote_margin_score(base, cands, metric):
    """Family decision = majority vote.  Confidence = vote-margin in [0,1]
    so the downstream conformal layer can use it instead of PRM."""
    out = _vote_majority(base, cands, metric)
    fp = out["final_prediction"]
    fp["aggregator"] = "vote_margin_score"
    if cands:
        fams = [c.get("family") for c in cands if c.get("parse_ok") and c.get("family")]
        if fams:
            counter = Counter(fams)
            top_n = counter.most_common(1)[0][1]
            second_n = counter.most_common(2)[1][1] if len(counter) > 1 else 0
            margin = (top_n - second_n) / max(len(cands), 1)
            fp["confidence"] = float(margin)
            fp["vote_top_n"] = int(top_n)
            fp["vote_second_n"] = int(second_n)
    return out


VARIANTS = {
    "greedy":                 lambda b, c, m: _greedy(b, c, m),
    "prm_argmax_n1":          lambda b, c, m: _prm_argmax(b, c, 1, m),
    "prm_argmax_n2":          lambda b, c, m: _prm_argmax(b, c, 2, m),
    "prm_argmax_n4":          lambda b, c, m: _prm_argmax(b, c, 4, m),
    "prm_argmax_n8":          lambda b, c, m: _prm_argmax(b, c, 8, m),
    "vote_majority":          lambda b, c, m: _vote_majority(b, c, m),
    "vote_then_prm_tiebreak": lambda b, c, m: _vote_then_prm_tiebreak(b, c, m),
    "vote_prm_weighted":      lambda b, c, m: _vote_prm_weighted(b, c, m),
    "prm_vote_consensus":     lambda b, c, m: _prm_vote_consensus(b, c, m),
    "vote_margin_score":      lambda b, c, m: _vote_margin_score(b, c, m),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--input_order", default="ab", choices=["ab", "ba"])
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--metric", default="geomean_plus",
                    choices=["geomean_plus", "min_plus", "mean_plus", "final_plus"])
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}.  Known: {sorted(VARIANTS)}")

    writers = {}
    counts = {v: 0 for v in variants}
    for v in variants:
        path = out_dir / f"pred_{args.run_name}_{v}_{args.input_order}.jsonl"
        writers[v] = path.open("w")
        print(f"[agg] writing {v:24s} -> {path}", flush=True)

    n_in = 0
    n_no_cands = 0
    try:
        for rec in _load_jsonl(Path(args.input)):
            n_in += 1
            cands = (rec.get("rerank") or {}).get("candidates") or []
            if not cands:
                n_no_cands += 1
            for v in variants:
                writers[v].write(json.dumps(VARIANTS[v](rec, cands, args.metric)) + "\n")
                counts[v] += 1
    finally:
        for w in writers.values():
            w.close()

    print(f"[agg] read {n_in:,} records  ({n_no_cands:,} had no candidates)")
    for v in variants:
        print(f"  {v:24s}  wrote {counts[v]:,} records")


if __name__ == "__main__":
    main()
