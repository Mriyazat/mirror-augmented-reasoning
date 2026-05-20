"""Mirror Family Stability (MFS): does the model predict the same family
for the AB and BA orderings of the same pair?

MFS = fraction of pairs where family(AB) == family(BA) AND not abstain.

This is a *structural* metric: tabular baselines have no notion of order
(they're a single feature vector), so MFS is undefined for them. Our LLM
sees the two drugs in a fixed order and has to be order-invariant by
*reasoning*, which is much harder. High MFS = the symmetry-KL training
worked.

Also computes MPS (Mirror Prediction Symmetry): exact final_prediction
agreement on all four fields (family, subtype, direction_tag, polarity).
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

SPLITS = {
    "random_full": ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_random_full_test_5000_stratified_nb_rerank8_abba.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_drug_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pred_phase4_prm_dpo_macro0797_pair_cold_test_5000_stratified_nb_rerank4_abba.jsonl",
}

MANIFESTS = {
    "random_full": ROOT / "outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl",
    "drug_cold":   ROOT / "outputs/eval_prompts/drug_cold_test_5000_stratified.manifest.jsonl",
    "pair_cold":   ROOT / "outputs/eval_prompts/pair_cold_test_5000_stratified.manifest.jsonl",
}


def _truth():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                         columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _manifest(path):
    if not path.exists():
        return set()
    out = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pair_id"])
    return out


def _load_ab_ba(path, keep):
    """Returns {pair_id -> (ab_pred, ba_pred)} where each pred is a dict
    or None if abstain / missing."""
    by = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in keep:
                continue
            order = (r.get("input_order") or "ab").lower()
            fp = r.get("final_prediction") or {}
            entry = {
                "family": fp.get("family"),
                "subtype": fp.get("subtype"),
                "direction_tag": fp.get("direction_tag"),
                "polarity": fp.get("polarity"),
                "abstain": bool(fp.get("abstain", False)),
            }
            by.setdefault(pid, {})[order] = entry
    out = {}
    for pid, d in by.items():
        out[pid] = (d.get("ab"), d.get("ba"))
    return out


def main():
    out_dir = ROOT / "outputs/diag2/headline"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    print(f"{'split':12s} {'n_paired':>8s} {'MFS':>8s} {'MPS':>8s} {'MFS_correct':>14s} {'family_mismatch':>16s}")
    print("-" * 80)

    for split, pred_path in SPLITS.items():
        keep = _manifest(MANIFESTS[split])
        if not pred_path.exists():
            print(f"[mfs] missing {pred_path}")
            continue
        ab_ba = _load_ab_ba(pred_path, keep)
        truth = _truth()

        n_paired = 0
        n_fam_match = 0
        n_full_match = 0
        n_fam_match_correct = 0  # both correct AND match
        family_mismatch_pairs = []

        for pid, (ab, ba) in ab_ba.items():
            if ab is None or ba is None:
                continue
            if ab["abstain"] or ba["abstain"]:
                continue
            if ab["family"] is None or ba["family"] is None:
                continue
            n_paired += 1
            if ab["family"] == ba["family"]:
                n_fam_match += 1
                gold = truth.get(pid)
                if gold is not None and ab["family"] == gold:
                    n_fam_match_correct += 1
            else:
                family_mismatch_pairs.append((pid, ab["family"], ba["family"]))
            if (ab["family"] == ba["family"]
                    and ab["subtype"] == ba["subtype"]
                    and ab["direction_tag"] == ba["direction_tag"]
                    and ab["polarity"] == ba["polarity"]):
                n_full_match += 1

        mfs = n_fam_match / max(1, n_paired)
        mps = n_full_match / max(1, n_paired)
        mfs_correct = n_fam_match_correct / max(1, n_fam_match)

        summary[split] = {
            "n_paired": n_paired,
            "MFS_family": mfs,
            "MPS_all_fields": mps,
            "MFS_correct_when_matched": mfs_correct,
            "n_family_mismatch": n_paired - n_fam_match,
        }
        print(f"{split:12s} {n_paired:>8d} {mfs:>8.4f} {mps:>8.4f} {mfs_correct:>14.4f} {n_paired - n_fam_match:>16d}")

        # Save a couple of family-mismatch examples for the appendix
        sample = family_mismatch_pairs[:10]
        summary[split]["mismatch_examples"] = [
            {"pair_id": p, "ab_family": a, "ba_family": b} for (p, a, b) in sample
        ]

    out = out_dir / "mfs_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
