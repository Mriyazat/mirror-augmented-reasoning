"""evaluation -- End-to-end evaluation harness.

One command, every metric, any split.

Inputs
------
    --predictions   JSONL file, one record per pair (per input ordering).
                    Each record:
                        {
                          "pair_id":      str,        # canonical (order-invariant)
                          "input_order":  "ab" | "ba",
                          "trace":        { ... rubric-format dict ... },
                          "context_ids":  list[str],  # retrieval bundle used
                          "final_prediction": {
                              "family":        str,
                              "subtype":       str,
                              "direction_tag": str,
                              "polarity":      str | null,
                              "abstain":       bool,
                              "confidence":    float | null,
                              "label_dist":    { family: prob, ... },  # optional
                          },
                          # Counterfactual / adversarial side predictions (optional):
                          "prediction_adv_ev":    { ... same shape ... },
                          "prediction_no_ev":     { ... same shape ... },
                          "prediction_perturbed": { ... same shape ... },
                          "perturbation":         str,    # for CfS only
                          "perturbation_relevant": bool,  # for CfS only
                        }

    --labels        parquet with pair_id, family, subtype, bidirectional,
                    subject_drugbank_id, a_id, b_id, polarity, severity.
                    Typically `data_processed/labels_hierarchical.parquet`
                    JOINed with `data_processed/pairs.parquet` for a_id/b_id
                    and the DDInter severity file for the severity column.

    --split         "random_full" | "drug_cold" | "pair_cold" | "subset25k"
                    Restricts scoring to pair_ids present in the split's
                    test manifest.  Optional -- if omitted, scores all
                    pairs in --predictions.

Outputs
-------
    outputs/results/<run_name>/
        metrics.json       -- every metric's full report dict
        metrics.md         -- human-readable summary (paper-ready tables)
        per_family.csv     -- per-family breakdown
        errors.jsonl       -- misclassified pairs for manual inspection

Design notes
------------
1.  **Legacy and baseline adapters.**  The harness expects the current trace schema.
    To evaluate legacy traces or baseline outputs (OpenDDI / zero-shot), a
    thin adapter (`_adapt_*` helpers) normalizes them into the canonical
    record shape before scoring.  This lets the same eval command
    produce apples-to-apples legacy-vs-current numbers.

2.  **HR vocab auto-loaded.**  If `--known_entities_file` is omitted,
    we call `src.metrics.hr.load_known_entities()` to pull from the
    DrugBank parquets automatically.

3.  **Stratified reports.**  All metrics that support per-family /
    per-severity breakdowns produce them.  The markdown report has
    three sections: overall, by family, by severity.

Usage
-----
    # Evaluate student predictions on subset25k test set:
    python -m src.evaluation.run_full_eval \
        --predictions outputs/student/ddi_v4_sft/predictions_test.jsonl \
        --labels      data_processed/labels_hierarchical.parquet \
        --split       subset25k \
        --run_name    v4_student_subset25k

    # Same harness, legacy comparison:
    python -m src.evaluation.run_full_eval \
        --predictions outputs/results/v3_baseline/v3_traces_adapted.jsonl \
        --labels      data_processed/labels_hierarchical.parquet \
        --run_name    v3_baseline \
        --trace_format v3
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    import pyarrow.parquet as pq
    _HAVE_PQ = True
except Exception:
    _HAVE_PQ = False

from src.metrics.au  import AbstentionRecord, au_single, au_curve
from src.metrics.cfs import CounterfactualRecord, cfs_corpus
from src.metrics.csa import CsaRecord, csa_corpus
from src.metrics.hr  import hr_corpus, load_known_entities
from src.metrics.mfs import mfs_corpus
from src.metrics.mps import MirrorRecord, mps_corpus
from src.metrics.rpc import rpc_corpus
from src.metrics.ris import RisRecord, ris_corpus
from src.metrics.slfs import slfs_corpus
from src.metrics.ths import Prediction, Truth, score_many


ROOT = Path(__file__).resolve().parents[2]


# ======================================================================
# Loaders
# ======================================================================
def _load_jsonl(path: str | Path) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _ensure_context_ids(recs: list[dict]) -> dict:
    """Rebuild missing/empty context_ids on prediction records via ContextBuilder.

    The v1 inference pipeline emitted predictions without context_ids, which
    silently zeroed MFS / HR / RIS (every evidence_id failed to resolve).
    Run this defensive pass on any record whose context_ids list is empty
    before computing metrics, and report the augmentation stats so the
    metrics.md captures the rebuild.

    Returns a dict {n_records, n_already, n_built, n_failed}.
    """
    n_already = 0
    n_built = 0
    n_failed = 0
    cache: dict[str, list[str]] = {}
    cb = None

    for r in recs:
        ids = r.get("context_ids") or []
        if ids:
            n_already += 1
            continue
        pid = r.get("pair_id") or ""
        if not pid:
            n_failed += 1
            continue
        if pid in cache:
            r["context_ids"] = cache[pid]
            n_built += 1
            continue
        if cb is None:
            try:
                from src.teacher.context_builder import ContextBuilder
                cb = ContextBuilder()
            except Exception as e:
                print(f"[eval] context auto-build disabled "
                      f"(ContextBuilder load failed: {e!r}). "
                      f"MFS/HR/RIS may be 0.")
                cb = False  # sentinel meaning "give up"
        if cb is False:
            n_failed += 1
            continue
        try:
            ids = sorted(cb.build(pid).context_ids())
        except Exception:
            ids = []
            n_failed += 1
        cache[pid] = ids
        if ids:
            r["context_ids"] = ids
            n_built += 1
        else:
            n_failed += 1
    return {
        "n_records": len(recs),
        "n_with_context_already": n_already,
        "n_context_built_here": n_built,
        "n_context_failed": n_failed,
    }


def _load_labels(path: str | Path) -> dict[str, dict]:
    if not _HAVE_PQ:
        raise RuntimeError("pyarrow not installed -- cannot read parquet labels file.")
    tbl = pq.read_table(path).to_pylist()
    return {r["pair_id"]: r for r in tbl}


def _load_split_pair_ids(split_manifest: str | Path, split_section: str) -> set[str] | None:
    """Load pair_ids for a split section.  Accepts either:

      - parquet: data_processed/splits/manifest_<split>.parquet  (current layout)
        columns: pair_id, split, family
      - json:    {"train": [...], "val": [...], "test": [...]}   (legacy/tests)
    """
    p = Path(split_manifest)
    if not p.exists():
        return None
    if p.suffix == ".parquet":
        if not _HAVE_PQ:
            return None
        tbl = pq.read_table(p, columns=["pair_id", "split"]).to_pylist()
        return {r["pair_id"] for r in tbl if r.get("split") == split_section}
    with open(p) as f:
        manifest = json.load(f)
    if split_section not in manifest:
        return None
    return set(manifest[split_section])


# ======================================================================
# Trace adapters (legacy / baseline -> current canonical shape)
# ======================================================================
def _adapt_v3_trace(v3_rec: dict) -> dict:
    """Adapt a legacy trace record into the current evaluator shape.

    Legacy records from earlier CoT-style DDI work had free-form prose rationales and a
    separate `pred_label` field.  We synthesize a minimal canonical trace:
      - Single-step trace with role="mechanism_of_action"
      - final_answer populated from the legacy prediction
      - context_ids empty (legacy records didn't emit them)
    This lets MFS / HR / RPC still score legacy outputs, albeit with
    weaker signal (no evidence IDs to resolve against).
    """
    return {
        "pair_id":       v3_rec.get("pair_id", ""),
        "input_order":   v3_rec.get("input_order", "ab"),
        "trace": {
            "steps": [{
                "step_id":      1,
                "role":         "mechanism_of_action",
                "claim":        v3_rec.get("rationale") or v3_rec.get("explanation") or "",
                "evidence_ids": [],
            }],
            "final_answer": {
                "family":        v3_rec.get("pred_family") or v3_rec.get("pred_label"),
                "subtype":       v3_rec.get("pred_subtype"),
                "direction_tag": v3_rec.get("pred_direction") or "bidirectional",
                "polarity":      v3_rec.get("pred_polarity"),
                "abstain":       bool(v3_rec.get("abstain", False)),
            },
        },
        "context_ids":    v3_rec.get("context_ids") or [],
        "final_prediction": {
            "family":        v3_rec.get("pred_family") or v3_rec.get("pred_label"),
            "subtype":       v3_rec.get("pred_subtype"),
            "direction_tag": v3_rec.get("pred_direction") or "bidirectional",
            "polarity":      v3_rec.get("pred_polarity"),
            "abstain":       bool(v3_rec.get("abstain", False)),
            "confidence":    v3_rec.get("confidence"),
            "label_dist":    v3_rec.get("label_dist") or {},
        },
    }


_ADAPTERS = {
    "v4": lambda r: r,
    "v3": _adapt_v3_trace,
}


# ======================================================================
# Metric-specific record builders
# ======================================================================
def _filter_and_adapt(recs: list[dict], adapter, split_pair_ids: set[str] | None) -> list[dict]:
    out: list[dict] = []
    for r in recs:
        ar = adapter(r)
        if split_pair_ids is not None and ar["pair_id"] not in split_pair_ids:
            continue
        out.append(ar)
    return out


def _mfs_inputs(preds: list[dict], labels: dict[str, dict]) -> list[dict]:
    return [
        {
            "trace":       p["trace"],
            "context_ids": p.get("context_ids") or [],
            "family":      labels.get(p["pair_id"], {}).get("family"),
        }
        for p in preds
    ]


def _rpc_inputs(preds: list[dict], labels: dict[str, dict]) -> list[dict]:
    return [
        {
            "trace":       p["trace"],
            "trace_id":    f"{p['pair_id']}:{p.get('input_order','ab')}",
            "gold_family": labels.get(p["pair_id"], {}).get("family"),
        }
        for p in preds
    ]


def _slfs_inputs(preds: list[dict], labels: dict[str, dict]) -> list[dict]:
    return [
        {"trace": p["trace"],
         "family": labels.get(p["pair_id"], {}).get("family")}
        for p in preds
    ]


def _hr_inputs(preds: list[dict], labels: dict[str, dict]) -> list[dict]:
    return [
        {"trace": p["trace"],
         "gold_family": labels.get(p["pair_id"], {}).get("family")}
        for p in preds
    ]


def _mps_inputs(preds: list[dict], labels: dict[str, dict]) -> list[MirrorRecord]:
    """Pair-up (ab, ba) predictions into MirrorRecords."""
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in preds:
        by_pair[p["pair_id"]][p.get("input_order", "ab")] = p["final_prediction"]

    out: list[MirrorRecord] = []
    for pid, sides in by_pair.items():
        if "ab" not in sides or "ba" not in sides:
            continue
        lab = labels.get(pid)
        if lab is None:
            continue
        a_id, b_id = lab.get("a_id"), lab.get("b_id")
        bidir = bool(lab.get("bidirectional", False))
        subj = lab.get("subject_drugbank_id")
        if bidir or subj is None:
            subj_side = "A"
        elif subj == a_id:
            subj_side = "A"
        elif subj == b_id:
            subj_side = "B"
        else:
            continue
        out.append(MirrorRecord(
            pair_id=pid, gold_family=lab["family"], gold_bidirectional=bidir,
            gold_subject_side=subj_side,
            pred_ab=sides["ab"], pred_ba=sides["ba"],
        ))
    return out


def _csa_inputs(preds: list[dict], labels: dict[str, dict]) -> list[CsaRecord]:
    by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in preds:
        by_pair[p["pair_id"]][p.get("input_order", "ab")] = p["final_prediction"]
    out: list[CsaRecord] = []
    for pid, sides in by_pair.items():
        if "ab" not in sides or "ba" not in sides:
            continue
        lab = labels.get(pid)
        if lab is None:
            continue
        out.append(CsaRecord(
            pair_id=pid, gold_family=lab["family"],
            pred_family_ab=sides["ab"].get("family"),
            pred_family_ba=sides["ba"].get("family"),
        ))
    return out


def _au_inputs(preds: list[dict], labels: dict[str, dict]) -> list[AbstentionRecord]:
    out: list[AbstentionRecord] = []
    for p in preds:
        lab = labels.get(p["pair_id"])
        if lab is None:
            continue
        fp = p["final_prediction"]
        correct = (fp.get("family") == lab.get("family"))
        out.append(AbstentionRecord(
            pair_id=p["pair_id"],
            pred_correct=correct,
            abstained=bool(fp.get("abstain", False)),
            confidence=fp.get("confidence"),
            gold_family=lab.get("family"),
            severity=lab.get("severity"),
        ))
    return out


def _ths_inputs(preds: list[dict], labels: dict[str, dict]):
    ths_preds, ths_truths = [], []
    for p in preds:
        lab = labels.get(p["pair_id"])
        if lab is None:
            continue
        fp = p["final_prediction"]
        if fp.get("abstain"):
            # abstention -- not a THS commitment; skip
            continue
        ths_preds.append(Prediction(
            family=fp.get("family") or "",
            subtype=fp.get("subtype") or "",
            direction=fp.get("direction_tag") or "bidirectional",
            polarity=fp.get("polarity"),
        ))
        ths_truths.append(Truth(
            family=lab.get("family") or "",
            subtype=lab.get("subtype") or "",
            bidirectional=bool(lab.get("bidirectional", False)),
            subject_drugbank_id=lab.get("subject_drugbank_id"),
            object_drugbank_id=lab.get("object_drugbank_id"),
            polarity=lab.get("polarity"),
            a_id=lab.get("a_id") or "",
            b_id=lab.get("b_id") or "",
        ))
    return ths_preds, ths_truths


def _cfs_inputs(preds: list[dict]) -> list[CounterfactualRecord]:
    out: list[CounterfactualRecord] = []
    for p in preds:
        if "prediction_perturbed" not in p:
            continue
        fp = p["final_prediction"]
        fp_pert = p["prediction_perturbed"]
        out.append(CounterfactualRecord(
            pair_id=p["pair_id"],
            perturbation=p.get("perturbation", "UNKNOWN"),
            relevant=bool(p.get("perturbation_relevant", False)),
            p_original=fp.get("label_dist") or {},
            p_perturbed=fp_pert.get("label_dist") or {},
        ))
    return out


def _ris_inputs(preds: list[dict], labels: dict[str, dict]) -> list[RisRecord]:
    out: list[RisRecord] = []
    for p in preds:
        if "prediction_adv_ev" not in p:
            continue
        lab = labels.get(p["pair_id"])
        if lab is None:
            continue
        fp = p["final_prediction"]
        fp_adv = p["prediction_adv_ev"]
        fp_no = p.get("prediction_no_ev") or {}
        out.append(RisRecord(
            pair_id=p["pair_id"],
            gold_label=lab.get("family") or "",
            pred_true_ev=fp.get("family") or "",
            pred_adv_ev=fp_adv.get("family") or "",
            pred_no_ev=fp_no.get("family") if fp_no else None,
            gold_family=lab.get("family"),
        ))
    return out


# ======================================================================
# Report writer
# ======================================================================
def _write_markdown(report: dict, out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# DDI Verifier evaluation -- {report['run_name']}\n")
    lines.append(f"- Predictions: `{report['predictions_path']}`")
    lines.append(f"- Labels:      `{report['labels_path']}`")
    lines.append(f"- Split:       `{report.get('split', 'all')}`")
    lines.append(f"- n pairs (after filter): {report['n_pairs_evaluated']:,}")
    lines.append("")
    lines.append("## Headline metrics\n")
    lines.append("| Metric | Value | Plan target |")
    lines.append("|---|---:|---:|")
    hl = report["headline"]
    lines.append(f"| **MFS**  | {hl['mfs']:.3f} | >= 0.70 |")
    lines.append(f"| **MPS**  | {hl['mps']:.3f} | >= 0.85 |")
    lines.append(f"| **CSA**  | {hl['csa']:.3f} | >= 0.80 |")
    lines.append(f"| **CfS gap**  | {hl['cfs_gap']:.3f} | >= 0.20 |")
    lines.append(f"| **AU @90%**  | {hl['au_at_90']:.3f} | > 0.0 |")
    lines.append(f"| **RPC**  | {hl['rpc']:.3f} | >= 0.90 |")
    lines.append(f"| **SLFS** | {hl['slfs']:.3f} | >= 0.70 |")
    lines.append(f"| **HR**   | {hl['hr']:.3f} | <= 0.10 |")
    lines.append(f"| **RIS**  | {hl['ris']:.3f} | > 0 |")
    lines.append(f"| **THS**  | {hl['ths']:.3f} | >= 0.65 |")
    lines.append(f"| **Family macro-F1** | {hl['family_macro_f1']:.3f} | - |")
    lines.append(f"| **Family accuracy** | {hl['family_accuracy']:.3f} | - |")
    lines.append("")

    # Per-family
    lines.append("## Per-family MFS / MPS / CSA / RPC / SLFS\n")
    lines.append("| Family | n | MFS | MPS | CSA | RPC | SLFS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    per_fam_all = report["per_family"]
    for fam in sorted(per_fam_all.keys()):
        row = per_fam_all[fam]
        lines.append(
            f"| `{fam}` | {row['n']} | "
            f"{row.get('mfs', 0):.2f} | "
            f"{row.get('mps', 0):.2f} | "
            f"{row.get('csa', 0):.2f} | "
            f"{row.get('rpc', 0):.2f} | "
            f"{row.get('slfs', 0):.2f} |"
        )
    lines.append("")

    out_md.write_text("\n".join(lines) + "\n")


_DDI_FAMILY_VOCAB = (
    "AdverseRisk",
    "Efficacy",
    "PD_Activity",
    "PK_Absorption",
    "PK_Distribution",
    "PK_Excretion",
    "PK_Metabolism",
)


def _family_accuracy(preds: list[dict], labels: dict[str, dict]):
    """Lightweight family-level macro-F1 + accuracy.  Implemented here to
    avoid pulling sklearn; matches the baseline_xgboost computation.

    Macro is computed over the canonical 7-family DDI vocabulary
    (`_DDI_FAMILY_VOCAB`).  Parse-failed predictions (where
    ``final_prediction.family`` is missing / None / not a known family)
    count as a false negative against their gold family but do NOT
    spawn an extra "None" / out-of-vocab class in the macro denominator.

    Pre-fix bug: predictions with ``family is None`` were incremented
    into ``fp[None]``, which added a phantom 8th class to the macro
    denominator with F1=0.  On a 7-family setup that cost ~12.5% of
    macro-F1 from a single missing field.  See git log for details.
    """
    from collections import Counter

    valid = set(_DDI_FAMILY_VOCAB)
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    total = right = 0
    for p in preds:
        lab = labels.get(p["pair_id"])
        if lab is None:
            continue
        fp_ = p["final_prediction"]
        if fp_.get("abstain"):
            continue
        gold = lab.get("family")
        pred = fp_.get("family")
        total += 1
        gold_in = gold in valid
        pred_in = pred in valid
        if pred_in and pred == gold:
            right += 1
            tp[gold] += 1
        else:
            if pred_in:
                fp[pred] += 1
            if gold_in:
                fn[gold] += 1
    f1s = []
    for f in _DDI_FAMILY_VOCAB:
        denom_p = tp[f] + fp[f]
        denom_r = tp[f] + fn[f]
        if denom_r == 0:
            continue
        if tp[f] == 0:
            f1s.append(0.0)
            continue
        prec = tp[f] / denom_p
        rec = tp[f] / denom_r
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    acc = right / total if total else 0.0
    return macro_f1, acc


# ======================================================================
# Main
# ======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels",      required=True)
    p.add_argument("--split", default=None,
                   choices=[None, "random_full", "drug_cold", "pair_cold", "subset25k"])
    p.add_argument("--split_section", default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--splits_dir", default="data_processed/splits")
    p.add_argument("--trace_format", default="v4", choices=["v4", "v3"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--known_entities_file", default=None,
                   help="JSON file with a list of strings; overrides DrugBank auto-load.")
    p.add_argument("--au_alpha", type=float, default=1.0)
    p.add_argument("--auto_context", dest="auto_context", action="store_true",
                   help="If a prediction record is missing context_ids, "
                        "rebuild it via ContextBuilder. Required for "
                        "MFS / HR / RIS to be meaningful. Default ON.")
    p.add_argument("--no_auto_context", dest="auto_context", action="store_false",
                   help="Disable the context rebuild fallback (matches old "
                        "behaviour; MFS/HR/RIS will be 0 on records with "
                        "empty context_ids).")
    p.set_defaults(auto_context=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir or (ROOT / "outputs" / "results" / args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    raw_preds = _load_jsonl(args.predictions)
    labels = _load_labels(args.labels)
    split_pair_ids = None
    if args.split:
        sp_dir = Path(args.splits_dir)
        manifest = sp_dir / f"manifest_{args.split}.parquet"
        if not manifest.exists():
            manifest = sp_dir / f"{args.split}.json"
        split_pair_ids = _load_split_pair_ids(manifest, args.split_section)
        if split_pair_ids is None:
            print(f"[warn] split manifest {manifest} not found or missing "
                  f"{args.split_section!r}; evaluating all predictions.")
    adapter = _ADAPTERS[args.trace_format]
    preds = _filter_and_adapt(raw_preds, adapter, split_pair_ids)
    print(f"[eval] n predictions after filter: {len(preds):,}")

    ctx_stats: dict | None = None
    if args.auto_context:
        n_missing = sum(1 for p in preds if not (p.get("context_ids") or []))
        if n_missing:
            print(f"[eval] auto-rebuilding context_ids for "
                  f"{n_missing:,}/{len(preds):,} records ...")
            ctx_stats = _ensure_context_ids(preds)
            print(f"[eval] context rebuild: built={ctx_stats['n_context_built_here']:,} "
                  f"failed={ctx_stats['n_context_failed']:,} "
                  f"already_present={ctx_stats['n_with_context_already']:,}")

    # Known-entity vocab for HR
    if args.known_entities_file:
        with open(args.known_entities_file) as f:
            known = set(json.load(f))
    else:
        try:
            known = load_known_entities()
            print(f"[eval] HR vocab auto-loaded: {len(known):,} entities")
        except Exception as e:
            print(f"[warn] HR vocab auto-load failed ({e}); "
                  f"HR will score all entities as hallucinated. "
                  f"Pass --known_entities_file to fix.")
            known = set()

    # Compute each metric
    mfs_rep  = mfs_corpus(_mfs_inputs(preds, labels))
    rpc_rep  = rpc_corpus(_rpc_inputs(preds, labels))
    slfs_rep = slfs_corpus(_slfs_inputs(preds, labels))
    hr_rep   = hr_corpus(_hr_inputs(preds, labels), known)

    mps_recs = _mps_inputs(preds, labels)
    mps_rep  = mps_corpus(mps_recs)
    csa_rep  = csa_corpus(_csa_inputs(preds, labels))

    au_recs  = _au_inputs(preds, labels)
    au_single_rep = au_single(au_recs, alpha=args.au_alpha)
    try:
        au_curve_rep = au_curve(au_recs, alpha=args.au_alpha)
    except Exception as e:
        au_curve_rep = {"error": str(e)}

    cfs_recs = _cfs_inputs(preds)
    cfs_rep  = cfs_corpus(cfs_recs) if cfs_recs else {"cfs_gap": 0.0, "n_relevant": 0}

    ris_recs = _ris_inputs(preds, labels)
    ris_rep  = ris_corpus(ris_recs) if ris_recs else {"ris": 0.0, "n": 0}

    ths_preds, ths_truths = _ths_inputs(preds, labels)
    ths_rep = score_many(ths_preds, ths_truths) if ths_preds else {"macro_ths": 0.0, "n": 0}

    macro_f1, acc = _family_accuracy(preds, labels)

    # Per-family rollup (union of metric breakdowns)
    per_family: dict[str, dict] = defaultdict(dict)
    for fam, v in mfs_rep.get("per_family_mfs", {}).items():
        per_family[fam]["mfs"] = v
        per_family[fam]["n"] = mfs_rep["per_family_n"].get(fam, 0)
    for fam, v in mps_rep.get("per_family_mps", {}).items():
        per_family[fam]["mps"] = v
    for fam, v in csa_rep.get("per_family_csa", {}).items():
        per_family[fam]["csa"] = v
    for fam, v in rpc_rep.get("per_family_rpc", {}).items():
        per_family[fam]["rpc"] = v
    for fam, v in slfs_rep.get("per_family_slfs", {}).items():
        per_family[fam]["slfs"] = v
    for fam in list(per_family):
        per_family[fam].setdefault("n", 0)

    # Assemble headline
    headline = {
        "mfs":              mfs_rep.get("macro_mfs", 0.0),
        "mps":              mps_rep.get("mps", 0.0),
        "csa":              csa_rep.get("csa", 0.0),
        "cfs_gap":          cfs_rep.get("cfs_gap", 0.0) if isinstance(cfs_rep, dict) else 0.0,
        "au_at_90":         au_curve_rep.get("au_at_90_coverage", au_single_rep.get("au", 0.0))
                             if isinstance(au_curve_rep, dict) else au_single_rep.get("au", 0.0),
        "rpc":              rpc_rep.get("macro_rpc", 0.0),
        "slfs":             slfs_rep.get("slfs", 0.0),
        "hr":               hr_rep.get("hr", 0.0),
        "ris":              ris_rep.get("ris", 0.0),
        "ths":              ths_rep.get("macro_ths", 0.0),
        "family_macro_f1":  macro_f1,
        "family_accuracy":  acc,
    }

    full = {
        "run_name":            args.run_name,
        "predictions_path":    args.predictions,
        "labels_path":         args.labels,
        "split":               args.split,
        "split_section":       args.split_section,
        "trace_format":        args.trace_format,
        "n_pairs_evaluated":   len(preds),
        "auto_context":        args.auto_context,
        "context_rebuild":     ctx_stats,
        "headline":            headline,
        "per_family":          dict(per_family),
        "mfs":                 mfs_rep,
        "mps":                 mps_rep,
        "csa":                 csa_rep,
        "cfs":                 cfs_rep,
        "au_single":           au_single_rep,
        "au_curve_summary":    {k: v for k, v in au_curve_rep.items() if k != "curve"}
                                if isinstance(au_curve_rep, dict) else {},
        "rpc":                 rpc_rep,
        "slfs":                slfs_rep,
        "hr":                  {k: v for k, v in hr_rep.items() if k != "unknown_top"},
        "hr_top_unknowns":     hr_rep.get("unknown_top", []),
        "ris":                 ris_rep,
        "ths":                 ths_rep,
    }

    (out_dir / "metrics.json").write_text(json.dumps(full, indent=2, default=str))
    _write_markdown(full, out_dir / "metrics.md")
    print(f"[eval] wrote {out_dir / 'metrics.json'}")
    print(f"[eval] wrote {out_dir / 'metrics.md'}")

    # Short console summary
    print()
    print(f"  MFS  = {headline['mfs']:.3f}   (target >= 0.70)")
    print(f"  MPS  = {headline['mps']:.3f}   (target >= 0.85)")
    print(f"  CSA  = {headline['csa']:.3f}   (target >= 0.80)")
    print(f"  RPC  = {headline['rpc']:.3f}   (target >= 0.90)")
    print(f"  SLFS = {headline['slfs']:.3f}   (target >= 0.70)")
    print(f"  HR   = {headline['hr']:.3f}   (target <= 0.10)")
    print(f"  THS  = {headline['ths']:.3f}   (target >= 0.65)")
    print(f"  Fam macro-F1 = {headline['family_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
