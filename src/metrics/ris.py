"""RIS -- Retrieval Influence Score.
Definition
----------
    RIS = Delta_acc(true-ev)  -  Delta_acc(adv-ev)
        = [ acc(with TRUE evidence)    - acc(no evidence) ]
        - [ acc(with ADV  evidence)    - acc(no evidence) ]
        = acc(true-ev)  -  acc(adv-ev)

Intuition
---------
A model can fail retrieval-robustness in two ways:
  (1) IGNORE true evidence:        acc(true) ~ acc(baseline).
  (2) BELIEVE fake evidence:       acc(adv) << acc(baseline).

RIS catches both at once:
  - Large positive RIS  -> true context helps AND adversarial context
                            is successfully rejected (good robustness).
  - Zero RIS            -> model doesn't distinguish true from adv
                            (context-blind OR equally gullible).
  - Negative RIS        -> adversarial context FOOLS the model more than
                            true context helps it (red flag).

Success criterion
-----------------
    RIS > 0 AND positive on adversarial subset (retrieval resistant).

Construction of adv evidence
----------------------------
Not this module's concern -- the caller supplies pre-built adversarial
contexts (see `src/data/build_adversarial_contexts.py` -- to be added).
Canonical adversarial perturbations for DDI:
  - "X does NOT inhibit CYP3A4" injected when X actually does.
  - Pathway IDs swapped between drug_A and drug_B.
  - Similarity metrics inverted (jaccard flipped to its complement).

Input data model
----------------
Three predictions per pair (true, adv, baseline):

    RisRecord(
        pair_id:            str,
        gold_label:         str,
        pred_true_ev:       str,          # pred under TRUE retrieved context
        pred_adv_ev:        str,          # pred under ADVERSARIAL context
        pred_no_ev:         str | None,   # optional: no-context baseline
        gold_family:        str | None,
    )

If pred_no_ev is None, RIS still computes but the Delta decomposition
is reported only for the true/adv half.

Usage
-----
    from src.metrics.ris import RisRecord, ris_corpus
    rep = ris_corpus(records)
    print(rep["ris"], rep["acc_true_ev"], rep["acc_adv_ev"])
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RisRecord:
    pair_id: str
    gold_label: str
    pred_true_ev: str
    pred_adv_ev: str
    pred_no_ev: str | None = None
    gold_family: str | None = None


def _acc(hits: int, total: int) -> float:
    return (hits / total) if total else 0.0


def ris_corpus(records: Iterable[RisRecord]) -> dict:
    """Compute RIS + the component accuracies + per-family breakdown."""
    n = 0
    n_true = 0
    n_adv = 0
    n_none = 0
    n_with_baseline = 0

    by_fam: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    # per-family: [n, n_true_ok, n_adv_ok, n_none_ok, n_with_baseline]

    for r in records:
        n += 1
        fam = r.gold_family or "UNKNOWN"
        by_fam[fam][0] += 1
        if r.pred_true_ev == r.gold_label:
            n_true += 1
            by_fam[fam][1] += 1
        if r.pred_adv_ev == r.gold_label:
            n_adv += 1
            by_fam[fam][2] += 1
        if r.pred_no_ev is not None:
            n_with_baseline += 1
            by_fam[fam][4] += 1
            if r.pred_no_ev == r.gold_label:
                n_none += 1
                by_fam[fam][3] += 1

    acc_true = _acc(n_true, n)
    acc_adv = _acc(n_adv, n)
    acc_none = _acc(n_none, n_with_baseline) if n_with_baseline else None
    ris = acc_true - acc_adv

    # Decomposition (optional -- only computed if baseline provided)
    delta_true = (acc_true - acc_none) if acc_none is not None else None
    delta_adv = (acc_adv - acc_none) if acc_none is not None else None

    # Per-family
    per_family: dict[str, dict] = {}
    for f, counts in by_fam.items():
        nt_f, true_f, adv_f, none_f, base_f = counts
        a_t = _acc(true_f, nt_f)
        a_a = _acc(adv_f, nt_f)
        a_n = _acc(none_f, base_f) if base_f else None
        per_family[f] = {
            "n":            nt_f,
            "acc_true_ev":  a_t,
            "acc_adv_ev":   a_a,
            "acc_no_ev":    a_n,
            "ris":          a_t - a_a,
        }

    return {
        "ris":                ris,
        "acc_true_ev":        acc_true,
        "acc_adv_ev":         acc_adv,
        "acc_no_ev":          acc_none,          # None if baseline not provided
        "delta_true":         delta_true,
        "delta_adv":          delta_adv,
        "n":                  n,
        "n_with_baseline":    n_with_baseline,
        "per_family_ris":     per_family,
        "adversarial_robust": (acc_adv >= acc_none) if acc_none is not None else None,
    }
