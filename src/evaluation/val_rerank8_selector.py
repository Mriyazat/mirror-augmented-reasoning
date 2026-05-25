"""evaluation -- Verifier rerank-8 selector evaluation.

Evaluates aggregation layers over Best-of-8 reranker candidates from the
trained student. Each input JSONL has eight parsed candidates per pair
together with PRM scores, the parsed reasoning trace, and the model's
chosen final prediction.

The script reports:

* **Fixed selectors** -- greedy, PRM argmax (geomean / mean / min / final),
  majority vote, vote weighted by log-PRM, and a trace-aware vote.
* **Candidate ceiling** -- oracle macro-F1 over the eight candidates.
* **Learned leave-one-split-out selectors** -- LogReg / Gradient Boosting /
  Random Forest trained on two splits and applied to the held-out split.
* **Within-split grouped CV selectors** -- pair-id-grouped K-fold inside a
  single split (the deployable setup).
* **Mirror reconciliation** -- AB / BA reconciliation by confidence after
  each learned selector.

Inputs are read from the validation manifests and the rerank-8 prediction
JSONLs; outputs are written under ``outputs/audit/verifier_rerank_selector/``.

Example
-------
    python -m src.evaluation.val_rerank8_selector \
        --run_prefix phase4_prm_dpo_macro0797 \
        --pred_dir   outputs/eval_prompts \
        --manifest_suffix val_5000_stratified

Both ``--run_prefix`` and ``--manifest_suffix`` are required to match
the file-naming convention used by the rerank-8 inference pass:
``pred_<run_prefix>_<split>_<manifest_suffix>_nb_rerank8_<ab|ba>.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.verifier_random_full_probe import FAMS, trace_majority, verifier_flags


ROOT = Path(__file__).resolve().parents[2]

SPLITS = ("random_full", "drug_cold", "pair_cold")
ORDERS = ("ab", "ba")

SCORE_FIELDS = ("prm_geomean", "prm_mean", "prm_min", "prm_final")

_TRACE_CACHE: dict[int, dict] = {}
_FEATURE_CACHE: dict[tuple[int, int, str, str], list[float]] = {}
_VERIFIER_CACHE: dict[int, list[float]] = {}
_TRACE_FAMILY_CACHE: dict[int, tuple[str | None, float, float]] = {}


@dataclass(frozen=True)
class Choice:
    family: str
    idx: int
    confidence: float
    source: str


@dataclass(frozen=True)
class IOLayout:
    pred_dir: Path
    manifest_dir: Path
    out_dir: Path
    run_prefix: str
    manifest_suffix: str

    def pred_path(self, split: str, order: str) -> Path:
        name = f"pred_{self.run_prefix}_{split}_{self.manifest_suffix}_nb_rerank8_{order}.jsonl"
        return self.pred_dir / name

    def manifest_path(self, split: str) -> Path:
        return self.manifest_dir / f"{split}_{self.manifest_suffix}.manifest.jsonl"


def load_truth(layout: IOLayout, split: str) -> dict[str, str]:
    truth: dict[str, str] = {}
    with layout.manifest_path(split).open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                truth[row["pair_id"]] = row["family"]
    return truth


def load_records(layout: IOLayout, split: str, order: str) -> list[dict]:
    recs: list[dict] = []
    with layout.pred_path(split, order).open() as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    return recs


def family_or_default(family: str | None) -> str:
    return family if family in FAMS else "PD_Activity"


def score_value(cand: dict, field: str = "prm_geomean") -> float:
    try:
        v = cand.get(field)
        return float(v) if v is not None and math.isfinite(float(v)) else 0.0
    except Exception:
        return 0.0


def log_score(cand: dict, field: str) -> float:
    return math.log10(max(score_value(cand, field), 1e-12))


def candidates(rec: dict) -> list[dict]:
    return (rec.get("rerank") or {}).get("candidates") or []


def parsed_trace(cand: dict) -> dict:
    key = id(cand)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    raw = cand.get("raw_output")
    if not isinstance(raw, str) or not raw.strip():
        _TRACE_CACHE[key] = {}
        return _TRACE_CACHE[key]
    try:
        obj = json.loads(raw)
    except Exception:
        _TRACE_CACHE[key] = {}
        return _TRACE_CACHE[key]
    _TRACE_CACHE[key] = obj if isinstance(obj, dict) else {}
    return _TRACE_CACHE[key]


def candidate_record(cand: dict) -> dict:
    """Minimal record shape consumed by the verifier feature extractor."""
    return {
        "trace": parsed_trace(cand),
        "final_prediction": {
            "family": cand.get("family"),
            "subtype": cand.get("subtype"),
            "direction_tag": cand.get("direction_tag"),
            "abstain": False,
            "confidence": score_value(cand, "prm_geomean"),
        },
    }


def trace_family_features(cand: dict) -> tuple[str | None, float, float]:
    key = id(cand)
    if key in _TRACE_FAMILY_CACHE:
        return _TRACE_FAMILY_CACHE[key]
    rec = candidate_record(cand)
    fam, strength = trace_majority(rec)
    agree = float(fam == cand.get("family")) if fam in FAMS else 0.0
    _TRACE_FAMILY_CACHE[key] = (fam, float(strength), agree)
    return _TRACE_FAMILY_CACHE[key]


def valid_candidates(rec: dict) -> list[tuple[int, dict]]:
    return [
        (i, c)
        for i, c in enumerate(candidates(rec))
        if c.get("parse_ok") and c.get("family") in FAMS
    ]


def choice_from_candidate(idx: int, cand: dict, source: str, confidence: float | None = None) -> Choice:
    conf = score_value(cand, "prm_geomean") if confidence is None else float(confidence)
    return Choice(family=family_or_default(cand.get("family")), idx=idx, confidence=conf, source=source)


def final_choice(rec: dict) -> Choice:
    fp = rec.get("final_prediction") or {}
    rr = rec.get("rerank") or {}
    return Choice(
        family=family_or_default(fp.get("family")),
        idx=int(rr.get("chosen_idx", -1) if rr.get("chosen_idx") is not None else -1),
        confidence=float(rr.get("chosen_prm") or fp.get("confidence") or 0.0),
        source="current_final",
    )


def greedy_choice(rec: dict) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    idx, cand = vc[0]
    return choice_from_candidate(idx, cand, "greedy")


def prm_argmax_choice(rec: dict, field: str) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    idx, cand = max(vc, key=lambda ic: (score_value(ic[1], field), -ic[0]))
    return choice_from_candidate(idx, cand, f"argmax_{field}", score_value(cand, field))


def family_votes(rec: dict) -> Counter:
    counter: Counter = Counter()
    for _, cand in valid_candidates(rec):
        counter[cand.get("family")] += 1
    return counter


def vote_majority_choice(rec: dict) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    votes = family_votes(rec)
    if not votes:
        return prm_argmax_choice(rec, "prm_geomean")
    top_n = votes.most_common(1)[0][1]
    top_fams = {fam for fam, n in votes.items() if n == top_n}
    fam_cands = [(i, c) for i, c in vc if c.get("family") in top_fams]
    idx, cand = max(fam_cands, key=lambda ic: (score_value(ic[1], "prm_geomean"), -ic[0]))
    confidence = top_n / max(len(vc), 1)
    return choice_from_candidate(idx, cand, "vote_majority", confidence)


def vote_prm_choice(rec: dict) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    fam_score: defaultdict[str, float] = defaultdict(float)
    for _, cand in vc:
        # Log-scores shifted to a positive range. Raw PRM scores are tiny and
        # make weighted votes numerically brittle.
        fam_score[cand.get("family")] += max(log_score(cand, "prm_geomean") + 12.0, 0.0)
    top_fam = max(fam_score, key=fam_score.get)
    fam_cands = [(i, c) for i, c in vc if c.get("family") == top_fam]
    idx, cand = max(fam_cands, key=lambda ic: (score_value(ic[1], "prm_geomean"), -ic[0]))
    total = sum(fam_score.values()) or 1.0
    return choice_from_candidate(idx, cand, "vote_prm_weighted", fam_score[top_fam] / total)


def trace_vote_choice(rec: dict) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    votes = family_votes(rec)
    for _, cand in vc:
        tfam, strength, agree = trace_family_features(cand)
        if tfam in FAMS:
            votes[tfam] += 0.85 * strength
        if agree:
            votes[cand.get("family")] += 0.35
    top_fam = max(votes, key=votes.get)
    fam_cands = [(i, c) for i, c in vc if c.get("family") == top_fam]
    if not fam_cands:
        return vote_prm_choice(rec)
    idx, cand = max(fam_cands, key=lambda ic: (score_value(ic[1], "prm_geomean"), -ic[0]))
    top = votes[top_fam]
    second = Counter(votes).most_common(2)[1][1] if len(votes) > 1 else 0.0
    confidence = (top - second) / max(sum(votes.values()), 1.0)
    return choice_from_candidate(idx, cand, "trace_vote", confidence)


FIXED_SELECTORS = {
    "current_final": final_choice,
    "greedy": greedy_choice,
    "argmax_prm_geomean": lambda r: prm_argmax_choice(r, "prm_geomean"),
    "argmax_prm_mean": lambda r: prm_argmax_choice(r, "prm_mean"),
    "argmax_prm_min": lambda r: prm_argmax_choice(r, "prm_min"),
    "argmax_prm_final": lambda r: prm_argmax_choice(r, "prm_final"),
    "vote_majority": vote_majority_choice,
    "vote_prm_weighted": vote_prm_choice,
    "trace_vote": trace_vote_choice,
}


def score_predictions(truth: dict[str, str], choices: dict[str, Choice]) -> dict:
    pids = sorted(set(truth) & set(choices))
    yt = [truth[pid] for pid in pids]
    yp = [choices[pid].family for pid in pids]
    return {
        "n": len(pids),
        "macro_f1": float(f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
    }


def evaluate_fixed(all_records: dict[tuple[str, str], list[dict]], truths: dict[str, dict[str, str]]) -> dict:
    results: dict[str, dict] = {}
    for name, selector in FIXED_SELECTORS.items():
        print(f"[selector] fixed {name}", flush=True)
        results[name] = {}
        for split in SPLITS:
            results[name][split] = {}
            for order in ORDERS:
                choices = {rec["pair_id"]: selector(rec) for rec in all_records[(split, order)]}
                results[name][split][order] = score_predictions(truths[split], choices)
            results[name][split]["mfs"] = mirror_stats(
                {rec["pair_id"]: selector(rec) for rec in all_records[(split, "ab")]},
                {rec["pair_id"]: selector(rec) for rec in all_records[(split, "ba")]},
                truths[split],
            )
    return results


def oracle_for_record(rec: dict, gold: str) -> Choice:
    vc = valid_candidates(rec)
    if not vc:
        return final_choice(rec)
    for idx, cand in vc:
        if cand.get("family") == gold:
            return choice_from_candidate(idx, cand, "oracle", 1.0)
    return prm_argmax_choice(rec, "prm_geomean")


def evaluate_oracle(all_records: dict[tuple[str, str], list[dict]], truths: dict[str, dict[str, str]]) -> dict:
    results: dict[str, dict] = {}
    for split in SPLITS:
        results[split] = {}
        for order in ORDERS:
            choices = {
                rec["pair_id"]: oracle_for_record(rec, truths[split][rec["pair_id"]])
                for rec in all_records[(split, order)]
            }
            results[split][order] = score_predictions(truths[split], choices)
        results[split]["mfs"] = mirror_stats(
            {
                rec["pair_id"]: oracle_for_record(rec, truths[split][rec["pair_id"]])
                for rec in all_records[(split, "ab")]
            },
            {
                rec["pair_id"]: oracle_for_record(rec, truths[split][rec["pair_id"]])
                for rec in all_records[(split, "ba")]
            },
            truths[split],
        )
    return results


def verifier_feature_vector(cand: dict) -> list[float]:
    key = id(cand)
    if key in _VERIFIER_CACHE:
        return _VERIFIER_CACHE[key]
    rec = candidate_record(cand)
    try:
        flags = verifier_flags(rec)
    except Exception:
        flags = {}
    _VERIFIER_CACHE[key] = [
        float(flags.get("violation_score") or 0.0),
        float(bool(flags.get("has_gap"))),
        float(bool(flags.get("has_neighbor"))),
        float(bool(flags.get("speculative_conclusion"))),
        float(bool(flags.get("gap_non_abstain"))),
        float(bool(flags.get("weak_gap_non_abstain"))),
        float(bool(flags.get("pk_metabolism_without_paired_cyp"))),
        float(bool(flags.get("pk_nonmetab_without_transport"))),
        float(bool(flags.get("adverse_from_gap"))),
        float(bool(flags.get("invented_neighbor"))),
        float(bool(flags.get("low_conf_non_abstain"))),
    ]
    return _VERIFIER_CACHE[key]


def candidate_features(rec: dict, idx: int, cand: dict, split: str, order: str) -> list[float]:
    cache_key = (id(rec), idx, split, order)
    if cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]
    vc = valid_candidates(rec)
    n = max(len(vc), 1)
    fam = cand.get("family")
    votes = family_votes(rec)
    top_vote = votes.most_common(1)[0][1] if votes else 0
    fam_vote = votes.get(fam, 0)
    vote_second = votes.most_common(2)[1][1] if len(votes) > 1 else 0

    ranks = {}
    for field in SCORE_FIELDS:
        ordered = sorted(vc, key=lambda ic: (score_value(ic[1], field), -ic[0]), reverse=True)
        ranks[field] = {i: rank for rank, (i, _) in enumerate(ordered)}

    fam_scores = [score_value(c, "prm_geomean") for _, c in vc if c.get("family") == fam]
    tfam, tstrength, tagree = trace_family_features(cand)
    raw = cand.get("raw_output") or ""
    trace = parsed_trace(cand)
    steps = trace.get("steps") if isinstance(trace, dict) else []
    steps = steps if isinstance(steps, list) else []

    feats = [
        idx / 7.0,
        float(order == "ba"),
        float(split == "random_full"),
        float(split == "drug_cold"),
        float(split == "pair_cold"),
        float(bool(cand.get("parse_ok"))),
        fam_vote / n,
        (fam_vote - vote_second) / n,
        top_vote / n,
        float(fam_vote == top_vote),
        tstrength,
        tagree,
        len(raw) / 5000.0,
        len(steps) / 10.0,
    ]
    feats.extend(score_value(cand, f) for f in SCORE_FIELDS)
    feats.extend(log_score(cand, f) for f in SCORE_FIELDS)
    feats.extend(1.0 / (1.0 + ranks[f].get(idx, 99)) for f in SCORE_FIELDS)
    feats.extend([
        max(fam_scores) if fam_scores else 0.0,
        float(np.mean(fam_scores)) if fam_scores else 0.0,
        float(np.std(fam_scores)) if len(fam_scores) > 1 else 0.0,
    ])
    feats.extend(float(fam == f) for f in FAMS)
    feats.extend(float(tfam == f) for f in FAMS)
    feats.extend(verifier_feature_vector(cand))
    _FEATURE_CACHE[cache_key] = feats
    return feats


def build_candidate_rows(
    all_records: dict[tuple[str, str], list[dict]],
    truths: dict[str, dict[str, str]],
    train_splits: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    y: list[int] = []
    for split in SPLITS:
        if split not in train_splits:
            continue
        truth = truths[split]
        for order in ORDERS:
            for rec in all_records[(split, order)]:
                gold = truth.get(rec["pair_id"])
                if gold not in FAMS:
                    continue
                for idx, cand in valid_candidates(rec):
                    rows.append(candidate_features(rec, idx, cand, split, order))
                    y.append(int(cand.get("family") == gold))
    return np.asarray(rows, dtype=float), np.asarray(y, dtype=int)


def build_split_candidate_rows(
    all_records: dict[tuple[str, str], list[dict]],
    truths: dict[str, dict[str, str]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str, int, str]]]:
    rows: list[list[float]] = []
    y: list[int] = []
    groups: list[str] = []
    meta: list[tuple[str, str, int, str]] = []
    truth = truths[split]
    for order in ORDERS:
        for rec in all_records[(split, order)]:
            pid = rec["pair_id"]
            gold = truth.get(pid)
            if gold not in FAMS:
                continue
            for idx, cand in valid_candidates(rec):
                rows.append(candidate_features(rec, idx, cand, split, order))
                y.append(int(cand.get("family") == gold))
                groups.append(pid)
                meta.append((pid, order, idx, family_or_default(cand.get("family"))))
    return (
        np.asarray(rows, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(groups),
        meta,
    )


def learned_choices(model, records: list[dict], split: str, order: str) -> dict[str, Choice]:
    out: dict[str, Choice] = {}
    for rec in records:
        vc = valid_candidates(rec)
        if not vc:
            out[rec["pair_id"]] = final_choice(rec)
            continue
        X = np.asarray([candidate_features(rec, idx, cand, split, order) for idx, cand in vc], dtype=float)
        prob = model.predict_proba(X)[:, 1]
        # Tie-break with vote+PRM priors while keeping the learned score dominant.
        ranked = []
        votes = family_votes(rec)
        for (idx, cand), p in zip(vc, prob):
            prior = 0.010 * votes.get(cand.get("family"), 0) + 0.002 * score_value(cand, "prm_geomean")
            ranked.append((float(p) + prior, idx, cand))
        conf, idx, cand = max(ranked, key=lambda t: (t[0], -t[1]))
        out[rec["pair_id"]] = choice_from_candidate(idx, cand, "learned_selector", conf)
    return out


def score_meta_predictions(
    truth: dict[str, str],
    meta: list[tuple[str, str, int, str]],
    probs: np.ndarray,
) -> tuple[dict[str, dict[str, Choice]], dict]:
    grouped: dict[tuple[str, str], list[tuple[float, int, str]]] = defaultdict(list)
    for (pid, order, idx, fam), prob in zip(meta, probs):
        grouped[(pid, order)].append((float(prob), idx, fam))

    choices: dict[str, dict[str, Choice]] = {order: {} for order in ORDERS}
    for (pid, order), items in grouped.items():
        prob, idx, fam = max(items, key=lambda t: (t[0], -t[1]))
        choices[order][pid] = Choice(family=fam, idx=idx, confidence=prob, source="cv_selector")

    scored = {
        order: score_predictions(truth, choices[order])
        for order in ORDERS
    }
    scored["mfs"] = mirror_stats(choices["ab"], choices["ba"], truth)
    rec_ab, rec_ba = reconcile_choices(choices["ab"], choices["ba"])
    scored["mirror_reconciled_ab"] = score_predictions(truth, rec_ab)
    scored["mirror_reconciled_ba"] = score_predictions(truth, rec_ba)
    scored["mirror_reconciled_mfs"] = mirror_stats(rec_ab, rec_ba, truth)
    return choices, scored


def mirror_stats(ab: dict[str, Choice], ba: dict[str, Choice], truth: dict[str, str]) -> dict:
    pids = sorted(set(ab) & set(ba) & set(truth))
    n_match = sum(1 for pid in pids if ab[pid].family == ba[pid].family)
    n_both_correct_match = sum(
        1
        for pid in pids
        if ab[pid].family == ba[pid].family == truth[pid]
    )
    return {
        "n_paired": len(pids),
        "mfs_family": n_match / max(len(pids), 1),
        "mfs_correct_when_matched": n_both_correct_match / max(n_match, 1),
        "n_family_mismatch": len(pids) - n_match,
    }


def reconcile_choices(ab: dict[str, Choice], ba: dict[str, Choice]) -> tuple[dict[str, Choice], dict[str, Choice]]:
    out_ab: dict[str, Choice] = {}
    out_ba: dict[str, Choice] = {}
    for pid in sorted(set(ab) & set(ba)):
        a, b = ab[pid], ba[pid]
        if a.family == b.family:
            out_ab[pid] = a
            out_ba[pid] = b
            continue
        chosen = a if a.confidence >= b.confidence else b
        reconciled = Choice(
            family=chosen.family,
            idx=chosen.idx,
            confidence=chosen.confidence,
            source=f"mirror_reconciled_from_{chosen.source}",
        )
        out_ab[pid] = reconciled
        out_ba[pid] = reconciled
    return out_ab, out_ba


def train_models(X: np.ndarray, y: np.ndarray) -> dict[str, object]:
    return {
        name: make_model(name).fit(X, y)
        for name in ("logreg", "gb", "rf")
    }


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.7),
        )
    if name == "gb":
        return GradientBoostingClassifier(
            random_state=17,
            n_estimators=140,
            learning_rate=0.045,
            max_depth=3,
            subsample=0.85,
        )
    if name == "rf":
        return RandomForestClassifier(
            random_state=17,
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(name)


def evaluate_learned(all_records: dict[tuple[str, str], list[dict]], truths: dict[str, dict[str, str]]) -> dict:
    results: dict[str, dict] = {}
    for heldout in SPLITS:
        train_splits = set(SPLITS) - {heldout}
        print(f"[selector] learned heldout={heldout} building train features", flush=True)
        X, y = build_candidate_rows(all_records, truths, train_splits)
        print(f"[selector] learned heldout={heldout} train candidates={len(y):,}", flush=True)
        models = train_models(X, y)
        results[heldout] = {
            "train_splits": sorted(train_splits),
            "n_train_candidates": int(len(y)),
            "train_positive_rate": float(y.mean()),
            "models": {},
        }
        for name, model in models.items():
            print(f"[selector] learned heldout={heldout} model={name}", flush=True)
            try:
                p = model.predict_proba(X)[:, 1]
                auc = float(roc_auc_score(y, p))
            except Exception:
                auc = float("nan")
            split_res: dict[str, dict] = {"train_auc": auc}
            choices_by_order = {}
            for order in ORDERS:
                choices = learned_choices(model, all_records[(heldout, order)], heldout, order)
                choices_by_order[order] = choices
                split_res[order] = score_predictions(truths[heldout], choices)
            split_res["mfs"] = mirror_stats(choices_by_order["ab"], choices_by_order["ba"], truths[heldout])
            rec_ab, rec_ba = reconcile_choices(choices_by_order["ab"], choices_by_order["ba"])
            split_res["mirror_reconciled_ab"] = score_predictions(truths[heldout], rec_ab)
            split_res["mirror_reconciled_ba"] = score_predictions(truths[heldout], rec_ba)
            split_res["mirror_reconciled_mfs"] = mirror_stats(rec_ab, rec_ba, truths[heldout])
            results[heldout]["models"][name] = split_res
    return results


def evaluate_within_split_cv(
    all_records: dict[tuple[str, str], list[dict]],
    truths: dict[str, dict[str, str]],
    n_splits: int = 5,
) -> dict:
    """Grouped CV by ``pair_id`` inside each validation split.

    Estimates the deployable setup: train the selector on a split's val
    predictions, then apply that trained selector to that split's test
    predictions. Grouping by pair_id keeps AB/BA candidates for the same
    pair in the same fold, avoiding mirror leakage.
    """
    out: dict[str, dict] = {}
    for split in SPLITS:
        print(f"[selector] within-split CV split={split} building rows", flush=True)
        X, y, groups, meta = build_split_candidate_rows(all_records, truths, split)
        out[split] = {
            "n_candidates": int(len(y)),
            "positive_rate": float(y.mean()),
            "models": {},
        }
        for model_name in ("logreg", "gb", "rf"):
            print(f"[selector] within-split CV split={split} model={model_name}", flush=True)
            probs = np.zeros(len(y), dtype=float)
            gkf = GroupKFold(n_splits=n_splits)
            aucs = []
            for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups), 1):
                model = make_model(model_name)
                model.fit(X[tr], y[tr])
                probs[va] = model.predict_proba(X[va])[:, 1]
                try:
                    aucs.append(float(roc_auc_score(y[va], probs[va])))
                except Exception:
                    pass
                print(
                    f"[selector] within-split CV split={split} model={model_name} fold={fold}/{n_splits}",
                    flush=True,
                )
            _, scored = score_meta_predictions(truths[split], meta, probs)
            scored["cv_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
            scored["cv_auc_std"] = float(np.std(aucs)) if aucs else float("nan")
            out[split]["models"][model_name] = scored
    return out


def best_rows(summary: dict) -> list[dict]:
    rows = []
    fixed = summary["fixed"]
    for selector, by_split in fixed.items():
        for split in SPLITS:
            rows.append({
                "family": "fixed",
                "selector": selector,
                "split": split,
                "ab_macro_f1": by_split[split]["ab"]["macro_f1"],
                "ba_macro_f1": by_split[split]["ba"]["macro_f1"],
                "mfs": by_split[split]["mfs"]["mfs_family"],
            })
    for split, held in summary["learned_loso"].items():
        for model, res in held["models"].items():
            rows.append({
                "family": "learned_loso",
                "selector": model,
                "split": split,
                "ab_macro_f1": res["ab"]["macro_f1"],
                "ba_macro_f1": res["ba"]["macro_f1"],
                "mfs": res["mfs"]["mfs_family"],
                "mirror_ab_macro_f1": res["mirror_reconciled_ab"]["macro_f1"],
            })
    for split, by_split in summary.get("within_split_cv", {}).items():
        for model, res in by_split["models"].items():
            rows.append({
                "family": "within_split_cv",
                "selector": model,
                "split": split,
                "ab_macro_f1": res["ab"]["macro_f1"],
                "ba_macro_f1": res["ba"]["macro_f1"],
                "mfs": res["mfs"]["mfs_family"],
                "mirror_ab_macro_f1": res["mirror_reconciled_ab"]["macro_f1"],
            })
    return rows


def write_markdown(out_dir: Path, summary: dict) -> None:
    lines = ["# Rerank-8 Selector Results\n\n"]
    lines.append("Scores are computed on the validation manifests. Learned selectors are evaluated leave-one-split-out.\n\n")

    lines.append("## Fixed selectors\n\n")
    lines.append("| Selector | Split | AB Macro-F1 | BA Macro-F1 | MFS |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for selector, by_split in summary["fixed"].items():
        for split in SPLITS:
            row = by_split[split]
            lines.append(
                f"| `{selector}` | `{split}` | {row['ab']['macro_f1']:.4f} | "
                f"{row['ba']['macro_f1']:.4f} | {row['mfs']['mfs_family']:.4f} |\n"
            )

    lines.append("\n## Candidate ceiling\n\n")
    lines.append("| Split | Oracle AB | Oracle BA | Oracle MFS |\n")
    lines.append("|---|---:|---:|---:|\n")
    for split in SPLITS:
        row = summary["oracle"][split]
        lines.append(
            f"| `{split}` | {row['ab']['macro_f1']:.4f} | {row['ba']['macro_f1']:.4f} | "
            f"{row['mfs']['mfs_family']:.4f} |\n"
        )

    if summary.get("learned_loso"):
        lines.append("\n## Learned leave-one-split-out selectors\n\n")
        lines.append("| Held-out split | Model | AB Macro-F1 | BA Macro-F1 | MFS | Mirror-reconciled AB |\n")
        lines.append("|---|---|---:|---:|---:|---:|\n")
        for split in SPLITS:
            for model, row in summary["learned_loso"][split]["models"].items():
                lines.append(
                    f"| `{split}` | `{model}` | {row['ab']['macro_f1']:.4f} | "
                    f"{row['ba']['macro_f1']:.4f} | {row['mfs']['mfs_family']:.4f} | "
                    f"{row['mirror_reconciled_ab']['macro_f1']:.4f} |\n"
                )

    lines.append("\n## Within-split grouped CV selectors\n\n")
    lines.append("| Split | Model | AB Macro-F1 | BA Macro-F1 | MFS | Mirror-reconciled AB | CV AUC |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|\n")
    for split in SPLITS:
        for model, row in summary.get("within_split_cv", {}).get(split, {}).get("models", {}).items():
            lines.append(
                f"| `{split}` | `{model}` | {row['ab']['macro_f1']:.4f} | "
                f"{row['ba']['macro_f1']:.4f} | {row['mfs']['mfs_family']:.4f} | "
                f"{row['mirror_reconciled_ab']['macro_f1']:.4f} | "
                f"{row['cv_auc_mean']:.3f} |\n"
            )

    rows = best_rows(summary)
    lines.append("\n## Best by split (AB)\n\n")
    lines.append("| Split | Selector type | Selector | AB Macro-F1 | MFS |\n")
    lines.append("|---|---|---|---:|---:|\n")
    for split in SPLITS:
        candidates_for_split = [r for r in rows if r["split"] == split]
        best = max(candidates_for_split, key=lambda r: r["ab_macro_f1"])
        lines.append(
            f"| `{split}` | `{best['family']}` | `{best['selector']}` | "
            f"{best['ab_macro_f1']:.4f} | {best['mfs']:.4f} |\n"
        )

    (out_dir / "summary.md").write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_prefix", required=True,
        help="Prefix of the rerank-8 prediction files (matches the trained student run name).",
    )
    ap.add_argument(
        "--manifest_suffix", default="val_5000_stratified",
        help="Manifest suffix used by both the truth manifest and the rerank-8 predictions.",
    )
    ap.add_argument(
        "--pred_dir", default=str(ROOT / "outputs" / "eval_prompts"),
        help="Directory containing the rerank-8 prediction JSONLs.",
    )
    ap.add_argument(
        "--manifest_dir", default=str(ROOT / "outputs" / "eval_prompts"),
        help="Directory containing the validation manifest JSONLs.",
    )
    ap.add_argument(
        "--out_dir", default=str(ROOT / "outputs" / "audit" / "verifier_rerank_selector"),
        help="Output directory for summary.json and summary.md.",
    )
    ap.add_argument(
        "--mode", choices=["full", "within_cv"], default="full",
        help="`within_cv` skips the slower leave-one-split-out table.",
    )
    args = ap.parse_args()

    layout = IOLayout(
        pred_dir=Path(args.pred_dir),
        manifest_dir=Path(args.manifest_dir),
        out_dir=Path(args.out_dir),
        run_prefix=args.run_prefix,
        manifest_suffix=args.manifest_suffix,
    )
    layout.out_dir.mkdir(parents=True, exist_ok=True)

    truths = {split: load_truth(layout, split) for split in SPLITS}
    all_records = {
        (split, order): load_records(layout, split, order)
        for split in SPLITS
        for order in ORDERS
    }
    summary = {
        "run_prefix": args.run_prefix,
        "manifest_suffix": args.manifest_suffix,
        "splits": list(SPLITS),
        "orders": list(ORDERS),
        "n_records": {f"{split}:{order}": len(all_records[(split, order)]) for split in SPLITS for order in ORDERS},
        "fixed": evaluate_fixed(all_records, truths),
        "oracle": evaluate_oracle(all_records, truths),
    }
    if args.mode == "full":
        summary["learned_loso"] = evaluate_learned(all_records, truths)
        summary["within_split_cv"] = evaluate_within_split_cv(all_records, truths)
        json_path = layout.out_dir / "summary.json"
        md_path = layout.out_dir / "summary.md"
    else:
        summary["learned_loso"] = {}
        summary["within_split_cv"] = evaluate_within_split_cv(all_records, truths)
        json_path = layout.out_dir / "within_split_cv_summary.json"
        md_path = layout.out_dir / "within_split_cv_summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    write_markdown(layout.out_dir, summary)
    if args.mode == "within_cv":
        md_path.write_text((layout.out_dir / "summary.md").read_text())
    print(f"[selector] wrote {json_path}")
    print(f"[selector] wrote {md_path}")

    print("\nBest fixed selectors by AB macro-F1:")
    for split in SPLITS:
        best_name, best_row = max(
            ((name, row[split]) for name, row in summary["fixed"].items()),
            key=lambda item: item[1]["ab"]["macro_f1"],
        )
        print(
            f"  {split:12s} {best_name:22s} "
            f"AB={best_row['ab']['macro_f1']:.4f} BA={best_row['ba']['macro_f1']:.4f} "
            f"MFS={best_row['mfs']['mfs_family']:.4f}"
        )
    if summary["learned_loso"]:
        print("\nLearned LOSO selectors:")
        for split in SPLITS:
            for model, row in summary["learned_loso"][split]["models"].items():
                print(
                    f"  heldout={split:12s} model={model:6s} "
                    f"AB={row['ab']['macro_f1']:.4f} BA={row['ba']['macro_f1']:.4f} "
                    f"MFS={row['mfs']['mfs_family']:.4f} "
                    f"mirrorAB={row['mirror_reconciled_ab']['macro_f1']:.4f}"
                )
    print("\nWithin-split grouped CV selectors:")
    for split in SPLITS:
        for model, row in summary["within_split_cv"][split]["models"].items():
            print(
                f"  split={split:12s} model={model:6s} "
                f"AB={row['ab']['macro_f1']:.4f} BA={row['ba']['macro_f1']:.4f} "
                f"MFS={row['mfs']['mfs_family']:.4f} "
                f"mirrorAB={row['mirror_reconciled_ab']['macro_f1']:.4f} "
                f"AUC={row['cv_auc_mean']:.3f}"
            )


if __name__ == "__main__":
    main()
