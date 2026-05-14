"""MPS -- Mirror-Pair Separation (novelty pillar P4).

Definition (plan §8)
--------------------
    MPS = 1 - P(mirror_error | correct_family)
        = P(direction prediction is correctly symmetric across order swap |
            family was predicted correctly in BOTH orderings)

Motivation -- the V3 failure this fixes
---------------------------------------
V3 (Qwen2.5-7B distilled) had 51.4 % of its errors attributable to
direction-mirror confusion: when the same pair was shown as (A, B) vs
(B, A), the student would copy its direction tag rather than flip it.
The model had learned pair-surface patterns, not the ordering-covariant
mechanism.  MPS was built to measure exactly this failure mode so that
ablations (e.g., -- Mirror DPO) can be attributed to it.

Why condition on "correct family"?
----------------------------------
If the model predicts a totally wrong family for either ordering, its
direction tag is not semantically interpretable (direction is only
meaningful relative to the mechanism).  Conditioning on family-correct
keeps MPS a clean measure of *symmetry awareness*, independent of
classifier accuracy.  The unconditional version (plus family accuracy)
can be derived at report time.

Data model
----------
For each pair we need predictions under BOTH input orderings (A, B) and
(B, A), plus gold-label info to know which direction_tag is symmetric.

    MirrorRecord(
        pair_id:              str,
        gold_family:          str,
        gold_bidirectional:   bool,           # True iff label is symmetric
        gold_subject_side:    "A" | "B",      # which drug is subject in
                                              # canonical (A, B) order
        pred_ab:              {family, direction_tag, ...},
        pred_ba:              {family, direction_tag, ...},
    )

Direction semantics (matches `src.metrics.ths`)
-----------------------------------------------
`direction_tag` is specified relative to the INPUT order:
    "a_to_b"        -- drug_1 in the input affects drug_2
    "b_to_a"        -- drug_2 in the input affects drug_1
    "bidirectional" -- symmetric / unordered label

When we swap (A, B) -> (B, A):
    - Bidirectional gold:  pred_ab and pred_ba should both be "bidirectional".
    - Directional gold:    pred_ab's tag should point to the gold subject from
                            drug_1, and pred_ba's tag should point to the gold
                            subject from drug_2.  If gold_subject_side == "A":
                                - (A, B)  -> drug_1 = A = subject -> "a_to_b"
                                - (B, A)  -> drug_2 = A = subject -> "b_to_a"
                            If gold_subject_side == "B":
                                - (A, B)  -> drug_2 = B = subject -> "b_to_a"
                                - (B, A)  -> drug_1 = B = subject -> "a_to_b"

Usage
-----
    from src.metrics.mps import MirrorRecord, mps_corpus
    recs = [MirrorRecord(...), ...]
    report = mps_corpus(recs)
    print(report["mps"], report["per_family_mps"])
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


# The canonical direction-tag vocabulary.  Any other value (including
# "n/a" for abstentions) is treated as *not correct* under MPS -- MPS
# measures symmetry of committed predictions, so abstentions don't
# contribute either way.  Abstention rates are reported separately by AU.
_DIRECTIONAL_TAGS = {"a_to_b", "b_to_a"}
_BIDIRECTIONAL_TAG = "bidirectional"


@dataclass(frozen=True)
class MirrorRecord:
    """One mirror-pair observation.  Both predictions required."""
    pair_id: str
    gold_family: str
    gold_bidirectional: bool
    gold_subject_side: str  # "A" (drug_1 is subject) or "B" (drug_2 is subject)
    pred_ab: dict
    pred_ba: dict


def _expected_direction(gold_bidirectional: bool, gold_subject_side: str,
                        input_order: str) -> str:
    """Return the direction tag a perfect model should predict.

    input_order:  "ab"  -> (drug_1, drug_2) = (A, B)
                  "ba"  -> (drug_1, drug_2) = (B, A)
    """
    if gold_bidirectional:
        return _BIDIRECTIONAL_TAG
    if gold_subject_side not in {"A", "B"}:
        raise ValueError(f"gold_subject_side must be 'A' or 'B', got {gold_subject_side!r}")

    # Directional case: subject must be drug_1 for "a_to_b", drug_2 for "b_to_a".
    if input_order == "ab":
        return "a_to_b" if gold_subject_side == "A" else "b_to_a"
    if input_order == "ba":
        return "b_to_a" if gold_subject_side == "A" else "a_to_b"
    raise ValueError(f"input_order must be 'ab' or 'ba', got {input_order!r}")


def _family_correct(rec: MirrorRecord) -> bool:
    """True iff both orderings predicted the gold family."""
    return (rec.pred_ab.get("family") == rec.gold_family
            and rec.pred_ba.get("family") == rec.gold_family)


def _direction_correct(rec: MirrorRecord) -> bool:
    """True iff the direction tags are correctly symmetric given the gold.

    Bidirectional gold:  both predictions must be "bidirectional".
    Directional gold:    the tag in (A, B) must point at the gold subject from
                         drug_1, and the tag in (B, A) must point at the gold
                         subject from drug_2 (i.e. the tag must FLIP correctly).
    """
    want_ab = _expected_direction(rec.gold_bidirectional, rec.gold_subject_side, "ab")
    want_ba = _expected_direction(rec.gold_bidirectional, rec.gold_subject_side, "ba")
    got_ab = rec.pred_ab.get("direction_tag")
    got_ba = rec.pred_ba.get("direction_tag")
    return got_ab == want_ab and got_ba == want_ba


def mps_corpus(records: Iterable[MirrorRecord]) -> dict:
    """Compute MPS + diagnostic breakdown over a corpus of mirror records.

    Returns:
        {
            "mps":                    float,           # main number
            "mps_all_pairs":          float,           # unconditional: direction correct / total
            "n_total":                int,
            "n_family_correct_both":  int,
            "n_direction_correct_given_family": int,
            "per_family_mps":         dict[str, float],
            "per_family_n":           dict[str, int],
            "per_direction_mps": {
                "bidirectional": {"n": int, "correct": int, "rate": float},
                "directional":   {"n": int, "correct": int, "rate": float},
            },
            "mirror_error_rate":      float,           # 1 - mps
        }

    If no records satisfy the family-correct-both condition, mps is NaN
    (reported as -1.0) because the question "given family correct, is
    direction symmetric?" is undefined.  Check n_family_correct_both > 0
    in the caller before citing the number in a paper.
    """
    recs = list(records)
    n_total = len(recs)
    n_fam_correct = 0
    n_dir_correct_cond = 0
    n_dir_correct_uncond = 0

    by_family_total: dict[str, int] = defaultdict(int)   # family-correct-both counts
    by_family_hit:   dict[str, int] = defaultdict(int)

    bidir_total = bidir_hit = 0
    dir_total = dir_hit = 0

    for r in recs:
        fam_ok = _family_correct(r)
        dir_ok = _direction_correct(r)
        if fam_ok:
            n_fam_correct += 1
            by_family_total[r.gold_family] += 1
            if dir_ok:
                n_dir_correct_cond += 1
                by_family_hit[r.gold_family] += 1
        if dir_ok:
            n_dir_correct_uncond += 1
        if r.gold_bidirectional:
            bidir_total += 1
            if dir_ok:
                bidir_hit += 1
        else:
            dir_total += 1
            if dir_ok:
                dir_hit += 1

    mps_main = (n_dir_correct_cond / n_fam_correct) if n_fam_correct else -1.0
    mps_all = (n_dir_correct_uncond / n_total) if n_total else -1.0

    per_family_mps = {
        f: (by_family_hit[f] / by_family_total[f])
        for f in by_family_total
    }

    return {
        "mps":                   mps_main,
        "mps_all_pairs":         mps_all,
        "n_total":               n_total,
        "n_family_correct_both": n_fam_correct,
        "n_direction_correct_given_family": n_dir_correct_cond,
        "per_family_mps":        per_family_mps,
        "per_family_n":          dict(by_family_total),
        "per_direction_mps": {
            "bidirectional": {
                "n": bidir_total, "correct": bidir_hit,
                "rate": (bidir_hit / bidir_total) if bidir_total else -1.0,
            },
            "directional": {
                "n": dir_total, "correct": dir_hit,
                "rate": (dir_hit / dir_total) if dir_total else -1.0,
            },
        },
        "mirror_error_rate":     (1.0 - mps_main) if mps_main >= 0 else -1.0,
    }


# ----------------------------------------------------------------------------
# Convenience helper: build MirrorRecord list from a flat predictions file.
# ----------------------------------------------------------------------------
def pair_predictions_to_mirror_records(
    predictions: Iterable[dict],
    labels_by_pair: dict[str, dict],
) -> list[MirrorRecord]:
    """Group a stream of per-order predictions into mirror records.

    Each prediction dict must contain:
        "pair_id":     str           canonical pair id (order-invariant)
        "input_order": "ab" | "ba"
        "family":      str
        "direction_tag": str
        (other fields ignored but passed through)

    `labels_by_pair[pair_id]` must contain:
        "family":            str
        "bidirectional":     bool
        "subject_drugbank_id": str or None
        "a_id":              str       drug_1 id in canonical (A, B) order
        "b_id":              str       drug_2 id in canonical (A, B) order

    Returns one MirrorRecord per pair_id for which both orderings are
    present in `predictions`.
    """
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in predictions:
        pid = p.get("pair_id")
        order = p.get("input_order")
        if pid is None or order not in {"ab", "ba"}:
            continue
        by_pair[pid][order] = p

    out: list[MirrorRecord] = []
    for pid, preds in by_pair.items():
        if "ab" not in preds or "ba" not in preds:
            continue
        lab = labels_by_pair.get(pid)
        if lab is None:
            continue
        a_id = lab["a_id"]; b_id = lab["b_id"]
        bidir = bool(lab.get("bidirectional", False))
        subj_id = lab.get("subject_drugbank_id")
        if bidir or subj_id is None:
            subj_side = "A"  # unused when bidir=True
        elif subj_id == a_id:
            subj_side = "A"
        elif subj_id == b_id:
            subj_side = "B"
        else:
            # Subject id doesn't match either a_id or b_id -- label is broken.
            continue
        out.append(MirrorRecord(
            pair_id=pid,
            gold_family=lab["family"],
            gold_bidirectional=bidir,
            gold_subject_side=subj_side,
            pred_ab=preds["ab"],
            pred_ba=preds["ba"],
        ))
    return out
