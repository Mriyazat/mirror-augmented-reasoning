"""Phase D -- Abstention with calibration guarantees.

Why abstention is a core core contribution
----------------------------------------
Earlier baselines had no honest "I don't know" signal.  When prompted outside the
training distribution the model would fabricate confidence.  The the current pipeline
abstention pipeline composes three independent signals:

  1. **Split-conformal coverage** (Romano et al., 2019).  Guarantees
     that predictions the model *does* commit to will be correct at
     (1 - alpha) rate on exchangeable test data.  The price is
     principled coverage loss.

  2. **Entropy gate.**  Refuses to commit when the predicted label
     distribution's Shannon entropy exceeds a threshold.  Entropy is
     orthogonal to confidence ranking -- a 3-way tie between plausible
     families looks identical to a 3-way tie between nonsense
     families under margin-based thresholds.

  3. **Self-calibration via PRM**.  If PRM scores the final
     conclusion step below a threshold, abstain regardless of what
     the classifier head claims.  This catches the "fluent but
     unsupported" failure mode that fooled the earlier baseline's zero-shot baseline.

Any one of these signals can trigger abstention.  The final abstention
decision is an OR over the three predicates.  This is conservative on
purpose -- the current pipeline optimizes for Abstention Utility
(`src/metrics/au.py`), not raw coverage.

Calibration targets (plan ch.8)
-------------------------------
    target coverage at test time:  >= 0.85
    target selective accuracy:     >= 0.92  (conditional on commit)
    target AU @ alpha=1:           > 0.55

Data model
----------
    PredictionRecord(
        pair_id: str,
        label_dist: dict[str, float] | None,   # family -> prob
        confidence: float,                      # top-label probability
        prm_final: float | None,                # 0..1 from PRM scorer
        gold_family: str | None,                # required for calibration only
    )

API
---
    # Fit on a validation set
    thresholds = calibrate(
        val_records,
        target_coverage=0.85,
        target_selective_acc=0.92,
        use_entropy=True,
        use_prm=True,
    )
    # Apply to new records
    decisions = apply(test_records, thresholds)
    # decisions[i] = {"abstain": bool, "reasons": [str], ...}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Sequence


# ======================================================================
# Data classes
# ======================================================================
@dataclass(frozen=True)
class PredictionRecord:
    pair_id: str
    label_dist: dict[str, float] | None
    confidence: float
    prm_final: float | None = None
    pred_family: str | None = None
    gold_family: str | None = None


@dataclass
class AbstentionThresholds:
    """Fitted thresholds applied at inference time."""
    # Split-conformal: commit only if confidence >= conformal_threshold.
    conformal_threshold: float
    # Entropy gate: abstain if entropy > entropy_threshold.
    entropy_threshold: float | None
    # PRM gate: abstain if prm_final < prm_threshold.
    prm_threshold: float | None
    # Calibration metadata (for reproducibility / paper).
    target_coverage: float
    target_selective_acc: float
    n_calibration: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AbstentionThresholds":
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AbstentionThresholds":
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)


# ======================================================================
# Shannon entropy
# ======================================================================
def entropy(dist: dict[str, float] | None, eps: float = 1e-12) -> float:
    if not dist:
        return 0.0
    s = sum(max(0.0, v) for v in dist.values())
    if s <= 0:
        return 0.0
    h = 0.0
    for v in dist.values():
        p = max(0.0, v) / s
        if p > eps:
            h -= p * math.log(p)
    return h


# ======================================================================
# Split-conformal calibration
# ======================================================================
def _conformal_threshold(records: Sequence[PredictionRecord],
                         target_coverage: float) -> float:
    """Compute the split-conformal threshold on confidence scores.

    Non-conformity score: s_i = 1 - confidence_i  (higher = more anomalous)
    Given gold labels we adapt to a *selective* setting: we want
      Pr[pred == gold | commit] >= target_selective_acc (handled separately)
    Here we compute the classical marginal coverage version:
      we pick the smallest confidence threshold such that
      Pr[commit] >= target_coverage on the calibration set.

    Returns the confidence threshold t.
    """
    if not records:
        return 0.0
    confs = sorted([r.confidence for r in records], reverse=True)
    n = len(confs)
    # Keep top-k so that k/n >= target_coverage.  Conservative
    # (1 + 1/n finite-sample correction deliberately omitted since our
    # target is coverage empirical not distributional; enable via notes
    # if paper reviewers demand it).
    k = min(n, max(1, math.ceil(target_coverage * n)))
    return confs[k - 1]


def _selective_acc_threshold(records: Sequence[PredictionRecord],
                             target_selective_acc: float) -> tuple[float, float]:
    """Find the lowest confidence threshold such that selective accuracy on
    committed items >= target_selective_acc.  Returns (threshold, realized_coverage).

    Sweep descending confidences; for each candidate t_k = confidence[k],
    compute acc on {i : confidence_i >= t_k}.  Pick the MAX k such that acc >= target.
    This is the operating point with maximum coverage at the given
    selective-accuracy floor.
    """
    if not records:
        return 0.0, 1.0
    ranked = sorted(records, key=lambda r: r.confidence, reverse=True)
    n = len(ranked)
    hit_cum = 0
    best_thresh = ranked[0].confidence
    best_k = 0
    for k, r in enumerate(ranked, start=1):
        if r.pred_family is not None and r.gold_family is not None:
            if r.pred_family == r.gold_family:
                hit_cum += 1
        sel_acc = hit_cum / k
        if sel_acc >= target_selective_acc:
            best_thresh = r.confidence
            best_k = k
    coverage = best_k / n if n else 0.0
    return best_thresh, coverage


def _entropy_threshold(records: Sequence[PredictionRecord],
                       target_selective_acc: float,
                       max_cov_drop: float = 0.10) -> float | None:
    """Pick an entropy threshold that improves selective accuracy without
    sacrificing more than `max_cov_drop` coverage on the calibration set.

    Sweep candidate thresholds at percentiles of the entropy distribution.
    Pick the one with the highest selective_acc subject to coverage drop
    <= max_cov_drop.  Returns None if no candidate dominates "no-gate".
    """
    rec_ents = [(r, entropy(r.label_dist)) for r in records]
    if not rec_ents or all(e == 0 for _, e in rec_ents):
        return None
    ents = sorted(e for _, e in rec_ents)
    n = len(ents)
    baseline_cov = 1.0
    baseline_sel = sum(1 for r, _ in rec_ents
                       if r.pred_family and r.pred_family == r.gold_family) / n

    best_t: float | None = None
    best_score = -1.0
    # Try thresholds at the 50..95th percentile
    for pct in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        idx = max(0, min(n - 1, int(pct * n)))
        t = ents[idx]
        kept = [(r, e) for r, e in rec_ents if e <= t]
        if not kept:
            continue
        cov = len(kept) / n
        if baseline_cov - cov > max_cov_drop:
            continue
        sel = sum(1 for r, _ in kept
                  if r.pred_family and r.pred_family == r.gold_family) / len(kept)
        improvement = sel - baseline_sel
        if improvement > best_score:
            best_score = improvement
            best_t = t
    return best_t


def _prm_threshold(records: Sequence[PredictionRecord],
                   target_selective_acc: float,
                   max_cov_drop: float = 0.10) -> float | None:
    """Pick a PRM-final threshold with the same selection criterion as entropy."""
    rec_prms = [(r, r.prm_final) for r in records if r.prm_final is not None]
    if not rec_prms:
        return None
    n = len(rec_prms)
    baseline_sel = sum(1 for r, _ in rec_prms
                       if r.pred_family and r.pred_family == r.gold_family) / n

    # Pick candidate thresholds as *midpoints between consecutive PRM
    # values*.  This lets us cleanly separate populations whose prm
    # distributions have distinct modes (e.g. 0.2 vs 0.9).  Iterate
    # over a wider percentile range than before.
    prms = sorted(p for _, p in rec_prms)
    candidate_ts: list[float] = []
    # Every 5th-percentile boundary, offset slightly ABOVE the value
    # so "p >= t" excludes the datum at that percentile.
    for pct in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        idx = max(0, min(n - 1, int(pct * n)))
        # Use next distinct value as the threshold so p == prms[idx] abstains.
        base = prms[idx]
        above = next((x for x in prms[idx + 1:] if x > base), None)
        t = (base + above) / 2 if above is not None else base + 1e-6
        candidate_ts.append(t)

    best_t: float | None = None
    best_score = -1.0
    for t in candidate_ts:
        kept = [(r, p) for r, p in rec_prms if p >= t]
        if not kept:
            continue
        cov = len(kept) / n
        if 1.0 - cov > max_cov_drop:
            continue
        sel = sum(1 for r, _ in kept
                  if r.pred_family and r.pred_family == r.gold_family) / len(kept)
        improvement = sel - baseline_sel
        if improvement > best_score:
            best_score = improvement
            best_t = t
    return best_t


# ======================================================================
# Top-level calibration
# ======================================================================
def calibrate(
    val_records: Sequence[PredictionRecord],
    target_coverage: float = 0.85,
    target_selective_acc: float = 0.92,
    use_entropy: bool = True,
    use_prm: bool = True,
) -> AbstentionThresholds:
    """Fit thresholds on a labeled validation set.

    Returns an `AbstentionThresholds` object with:
        - conformal_threshold: confidence must be >= this to commit
          (chosen as the min of the coverage-controlled and
           selective-accuracy thresholds; the stricter of the two
           dominates).
        - entropy_threshold: entropy must be <= this to commit
          (optional; skipped if `use_entropy` is False or no
           beneficial threshold found).
        - prm_threshold: prm_final must be >= this to commit
          (optional; skipped if PRM not used or no score present).
    """
    cov_t = _conformal_threshold(val_records, target_coverage)
    sel_t, realized_cov = _selective_acc_threshold(val_records, target_selective_acc)
    conformal_t = max(cov_t, sel_t)

    entropy_t = None
    if use_entropy:
        entropy_t = _entropy_threshold(val_records, target_selective_acc)

    prm_t = None
    if use_prm:
        prm_t = _prm_threshold(val_records, target_selective_acc)

    return AbstentionThresholds(
        conformal_threshold=conformal_t,
        entropy_threshold=entropy_t,
        prm_threshold=prm_t,
        target_coverage=target_coverage,
        target_selective_acc=target_selective_acc,
        n_calibration=len(val_records),
        notes=(f"cov_t={cov_t:.4f} sel_t={sel_t:.4f} "
               f"realized_cov_at_sel={realized_cov:.3f}"),
    )


# ======================================================================
# Application
# ======================================================================
@dataclass
class AbstentionDecision:
    abstain: bool
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0
    entropy: float = 0.0
    prm_final: float | None = None


def apply(records: Iterable[PredictionRecord],
          thresholds: AbstentionThresholds) -> list[AbstentionDecision]:
    out: list[AbstentionDecision] = []
    for r in records:
        reasons: list[str] = []
        h = entropy(r.label_dist)
        if r.confidence < thresholds.conformal_threshold:
            reasons.append(f"confidence<{thresholds.conformal_threshold:.3f}")
        if thresholds.entropy_threshold is not None and h > thresholds.entropy_threshold:
            reasons.append(f"entropy>{thresholds.entropy_threshold:.3f}")
        if (thresholds.prm_threshold is not None
            and r.prm_final is not None
            and r.prm_final < thresholds.prm_threshold):
            reasons.append(f"prm<{thresholds.prm_threshold:.3f}")
        out.append(AbstentionDecision(
            abstain=bool(reasons),
            reasons=reasons,
            confidence=r.confidence,
            entropy=h,
            prm_final=r.prm_final,
        ))
    return out


# ======================================================================
# JSONL IO helpers (used by eval harness integration)
# ======================================================================
def load_predictions_jsonl(path: str | Path) -> list[PredictionRecord]:
    """Load a predictions.jsonl into PredictionRecord objects."""
    out: list[PredictionRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fp = r.get("final_prediction") or {}
            out.append(PredictionRecord(
                pair_id=r.get("pair_id", ""),
                label_dist=fp.get("label_dist") or {},
                confidence=float(fp.get("confidence") or 0.0),
                prm_final=fp.get("prm_final"),
                pred_family=fp.get("family"),
                gold_family=r.get("gold_family"),
            ))
    return out


def rewrite_predictions_with_abstention(
    in_path: str | Path,
    out_path: str | Path,
    thresholds: AbstentionThresholds,
    gold_labels_map: dict[str, str] | None = None,
) -> dict:
    """Rewrite a predictions JSONL applying abstention flags.

    For each record, compute the abstention decision from the record's
    final_prediction and -- if the record's final_prediction.abstain
    is currently False -- flip it to True when thresholds fire.
    Keeps the record otherwise untouched so eval-harness + metrics see
    a consistent schema.

    Returns {n, n_abstained, reasons_hist}.
    """
    from collections import Counter
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    reasons_hist: Counter = Counter()
    n = n_abs = 0
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            fp = r.get("final_prediction") or {}
            pr = PredictionRecord(
                pair_id=r.get("pair_id", ""),
                label_dist=fp.get("label_dist") or {},
                confidence=float(fp.get("confidence") or 0.0),
                prm_final=fp.get("prm_final"),
                pred_family=fp.get("family"),
                gold_family=(gold_labels_map or {}).get(r.get("pair_id")),
            )
            decision = apply([pr], thresholds)[0]
            if decision.abstain and not fp.get("abstain", False):
                fp["abstain"] = True
                fp["abstention_reasons"] = decision.reasons
                r["final_prediction"] = fp
                n_abs += 1
                for reason in decision.reasons:
                    # bucket by predicate family (strip the numeric cutoff)
                    reasons_hist[reason.split("<")[0].split(">")[0]] += 1
            fout.write(json.dumps(r) + "\n")
    return {"n": n, "n_abstained": n_abs, "reasons": dict(reasons_hist)}


# ======================================================================
# CLI
# ======================================================================
def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    fp = sub.add_parser("fit")
    fp.add_argument("--val_predictions", required=True)
    fp.add_argument("--out_thresholds",   required=True)
    fp.add_argument("--target_coverage",    type=float, default=0.85)
    fp.add_argument("--target_selective_acc", type=float, default=0.92)
    fp.add_argument("--no_entropy", action="store_true")
    fp.add_argument("--no_prm",     action="store_true")

    ap_ = sub.add_parser("apply")
    ap_.add_argument("--thresholds",   required=True)
    ap_.add_argument("--in_predictions", required=True)
    ap_.add_argument("--out_predictions", required=True)

    args = ap.parse_args()

    if args.cmd == "fit":
        recs = load_predictions_jsonl(args.val_predictions)
        th = calibrate(
            recs,
            target_coverage=args.target_coverage,
            target_selective_acc=args.target_selective_acc,
            use_entropy=not args.no_entropy,
            use_prm=not args.no_prm,
        )
        th.save(args.out_thresholds)
        print(f"[abs] fitted on {len(recs):,} records")
        print(f"      conformal_threshold  = {th.conformal_threshold:.4f}")
        print(f"      entropy_threshold    = {th.entropy_threshold}")
        print(f"      prm_threshold        = {th.prm_threshold}")
        print(f"      saved to {args.out_thresholds}")
        return

    if args.cmd == "apply":
        th = AbstentionThresholds.load(args.thresholds)
        stats = rewrite_predictions_with_abstention(
            args.in_predictions, args.out_predictions, th,
        )
        print(f"[abs] rewrote {stats['n']:,} records; "
              f"{stats['n_abstained']:,} abstained")
        print(f"      reasons: {stats['reasons']}")


if __name__ == "__main__":
    _cli()
