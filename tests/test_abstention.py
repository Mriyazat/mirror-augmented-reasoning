"""Smoke tests for the abstention module."""
from __future__ import annotations

import random

from src.inference.abstention import (
    PredictionRecord, calibrate, apply, entropy, AbstentionThresholds,
)


def _mk(conf: float, gold: str, pred: str,
        dist: dict[str, float] | None = None,
        prm: float | None = None) -> PredictionRecord:
    return PredictionRecord(
        pair_id=f"P{random.randint(0, 1_000_000)}",
        label_dist=dist or {pred: conf, "OTHER": 1.0 - conf},
        confidence=conf,
        prm_final=prm,
        pred_family=pred,
        gold_family=gold,
    )


def test_entropy_zero_and_max():
    assert entropy({"A": 1.0}) < 1e-6
    assert abs(entropy({"A": 0.5, "B": 0.5}) - 0.6931) < 1e-3
    # Unnormalized input -> same entropy after normalization
    assert abs(entropy({"A": 5.0, "B": 5.0}) - 0.6931) < 1e-3


def test_conformal_marginal_coverage():
    # 100 records with uniform confidences; target 80% coverage -> threshold ~0.2
    rng = random.Random(0)
    recs = []
    for _ in range(100):
        c = rng.random()
        gold = "PK"
        pred = "PK" if c > 0.3 else "PD"
        recs.append(_mk(c, gold, pred))
    th = calibrate(recs, target_coverage=0.8, target_selective_acc=0.5,
                   use_entropy=False, use_prm=False)
    # Target: keep top 80% -> threshold ~= 0.2 (or the 20th percentile conf)
    assert 0.10 < th.conformal_threshold < 0.35, th.conformal_threshold


def test_selective_accuracy_raises_threshold():
    # Wrong predictions concentrated at low confidences; target 95% selective acc
    # should push threshold above the wrong examples.
    recs = []
    # 50 correct with high conf
    for _ in range(50):
        recs.append(_mk(0.9, "PK", "PK"))
    # 50 wrong with low conf
    for _ in range(50):
        recs.append(_mk(0.4, "PK", "PD"))
    th = calibrate(recs, target_coverage=0.5, target_selective_acc=0.95,
                   use_entropy=False, use_prm=False)
    # Threshold should be > 0.4 so wrong preds are filtered
    assert th.conformal_threshold > 0.4


def test_entropy_gate_helps_when_wrong_has_high_entropy():
    # Correct preds have peaked distribution; wrong preds flat (high entropy).
    # The entropy threshold should flag the flat ones.
    correct = [
        PredictionRecord(pair_id=str(i), label_dist={"PK": 0.9, "PD": 0.1},
                         confidence=0.9, prm_final=None,
                         pred_family="PK", gold_family="PK")
        for i in range(50)
    ]
    wrong = [
        PredictionRecord(pair_id=str(i + 100), label_dist={"PK": 0.34, "PD": 0.33, "RX": 0.33},
                         confidence=0.34, prm_final=None,
                         pred_family="PK", gold_family="PD")
        for i in range(50)
    ]
    th = calibrate(correct + wrong, target_coverage=0.7,
                   target_selective_acc=0.90,
                   use_entropy=True, use_prm=False)
    # Entropy threshold should be picked to be between peaked and flat.
    # Peaked entropy ~= 0.325, flat entropy ~= 1.098.  Threshold should be in between.
    assert th.entropy_threshold is not None
    assert 0.3 < th.entropy_threshold < 1.1, th.entropy_threshold


def test_prm_gate_helps():
    # Correct preds have high PRM; a small set of wrong preds have low PRM.
    # Calibration's max_cov_drop=0.10 so our bad set must be <= 10%.
    correct = [
        _mk(0.6, "PK", "PK", prm=0.9) for _ in range(90)
    ]
    wrong = [
        _mk(0.6, "PK", "PD", prm=0.2) for _ in range(10)
    ]
    th = calibrate(correct + wrong, target_coverage=0.5,
                   target_selective_acc=0.95,
                   use_entropy=False, use_prm=True)
    # PRM threshold should filter out the wrong (low-prm) records
    assert th.prm_threshold is not None, th
    assert th.prm_threshold > 0.2, th.prm_threshold
    decisions = apply(correct + wrong, th)
    n_abs_correct = sum(1 for d, r in zip(decisions, correct + wrong)
                        if d.abstain and r.gold_family == r.pred_family)
    n_abs_wrong   = sum(1 for d, r in zip(decisions, correct + wrong)
                        if d.abstain and r.gold_family != r.pred_family)
    # Most wrong preds should be abstained; very few correct should be.
    assert n_abs_wrong >= 8, n_abs_wrong
    assert n_abs_correct < 5, n_abs_correct


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "th.json"
        th = AbstentionThresholds(
            conformal_threshold=0.5, entropy_threshold=0.8,
            prm_threshold=0.3, target_coverage=0.85,
            target_selective_acc=0.92, n_calibration=500,
            notes="test",
        )
        th.save(p)
        th2 = AbstentionThresholds.load(p)
        assert th2 == th


if __name__ == "__main__":
    random.seed(42)
    test_entropy_zero_and_max()
    test_conformal_marginal_coverage()
    test_selective_accuracy_raises_threshold()
    test_entropy_gate_helps_when_wrong_has_high_entropy()
    test_prm_gate_helps()
    test_save_load_roundtrip()
    print("ABSTENTION TESTS OK")
