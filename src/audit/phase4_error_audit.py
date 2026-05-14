"""Phase 4.1 -- audit Phase 3 student errors before targeted regeneration.

This script is intentionally conservative about file integrity. We previously
lost time by training/evaluating against a stale Phase-C corpus, so this audit
prints hashes, line counts, pair-id coverage, and fails hard when the prediction
file and validation file do not refer to the same pair set.

Inputs
------
  --predictions_ab   Phase 3.4 AB predictions JSONL.
  --predictions_ba   Optional BA predictions JSONL.
  --val_file         Correct Phase-C validation JSONL used for prediction.

Outputs
-------
  <output_dir>/phase4_error_audit.md
  <output_dir>/phase4_error_audit.json
  <output_dir>/phase4_error_examples.jsonl
  <output_dir>/phase4_integrity.json

The cluster labels are heuristic by design. They are not final scientific
claims; they identify where targeted teacher-regeneration prompts should focus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILIES = (
    "AdverseRisk",
    "Efficacy",
    "PD_Activity",
    "PK_Absorption",
    "PK_Distribution",
    "PK_Excretion",
    "PK_Metabolism",
)

ADVERSE_SUBTYPE_TO_PD = {
    "hypoglycemia": "hypoglycemic",
    "hyperkalemia": "hyperkalemic",
    "bradycardia": "bradycardic",
    "hypotension": "hypotensive",
    "hypertension": "hypertensive",
    "qtc_prolongation": "qtc_prolonging",
    "bleeding": "anticoagulant",
    "bleeding_and_hemorrhage": "anticoagulant",
    "cns_depression": "central_nervous_system_depressant_cns_depressant",
    "serotonin_syndrome": "serotonergic",
}

PK_METABOLISM_CUES = (
    "cyp", "cytochrome", "ugt", "enzyme", "metabol", "substrate",
    "inhibitor", "inhibits", "inducer", "induces",
)
PK_EXCRETION_CUES = (
    "renal", "kidney", "excretion", "clearance", "oat", "oct", "oatp",
    "bcrp", "biliary", "transporter",
)
PK_DISTRIBUTION_CUES = (
    "albumin", "glycoprotein", "protein binding", "serum concentration",
    "plasma concentration", "carrier", "displace", "binding",
)
EFFICACY_CUES = (
    "therapeutic efficacy", "efficacy", "antineoplastic", "antiviral",
    "effectiveness", "response", "activity", "agonist", "antagonist",
)
PD_CUES = (
    "additive", "synergy", "synergistic", "antagon", "receptor",
    "pharmacodynamic", "combined effect", "hypogly", "hyperkal",
    "brady", "qtc", "sedat", "seroton", "blood pressure",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"[audit] JSON parse error in {path}:{ln}: {e}") from e
    return rows


def _integrity(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pids = [r.get("pair_id") for r in rows]
    return {
        "path": str(path),
        "line_count": len(rows),
        "sha256": _sha256(path),
        "first_pair_id": pids[0] if pids else None,
        "last_pair_id": pids[-1] if pids else None,
        "unique_pair_ids": len(set(pids)),
        "duplicate_pair_ids": len(pids) - len(set(pids)),
    }


def _index_by_pair(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    dupes: list[str] = []
    for r in rows:
        pid = r.get("pair_id")
        if not pid:
            raise SystemExit(f"[audit] {name} contains a row without pair_id")
        if pid in out:
            dupes.append(pid)
        out[pid] = r
    if dupes:
        raise SystemExit(
            f"[audit] {name} has duplicate pair_ids (first 10): {dupes[:10]}"
        )
    return out


def _normalize_subtype(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    replacements = {
        "hyperkalemia": "hyperkalem",
        "hyperkalemic": "hyperkalem",
        "hypoglycemia": "hypoglycem",
        "hypoglycemic": "hypoglycem",
        "bradycardia": "bradycard",
        "bradycardic": "bradycard",
        "hypertension": "hypertens",
        "hypertensive": "hypertens",
        "hypotension": "hypotens",
        "hypotensive": "hypotens",
        "qtc_prolongation": "qtc_prolong",
        "qtc_prolonging": "qtc_prolong",
    }
    return replacements.get(s, s)


def _token_set(s: str | None) -> set[str]:
    if not s:
        return set()
    return {
        t for t in re.split(r"[^a-zA-Z0-9]+", s.lower())
        if len(t) >= 4
    }


def _subtype_equivalent(gold_sub: str | None, pred_sub: str | None) -> bool:
    g = _normalize_subtype(gold_sub)
    p = _normalize_subtype(pred_sub)
    if not g or not p:
        return False
    if g == p or g in p or p in g:
        return True
    return bool(_token_set(gold_sub) & _token_set(pred_sub))


def _assistant_text_from_val(rec: dict[str, Any]) -> str:
    for m in rec.get("messages") or []:
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def _user_prompt_from_val(rec: dict[str, Any]) -> str:
    for m in rec.get("messages") or []:
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _trace_text(pred: dict[str, Any], val: dict[str, Any]) -> str:
    pieces: list[str] = []
    fp = pred.get("final_prediction") or {}
    pieces.append(str(fp.get("family") or ""))
    pieces.append(str(fp.get("subtype") or ""))
    trace = pred.get("trace") or {}
    for step in trace.get("steps") or []:
        pieces.append(str(step.get("role") or ""))
        pieces.append(str(step.get("family_hint") or ""))
        pieces.append(str(step.get("claim") or ""))
        pieces.extend(str(x) for x in (step.get("evidence_ids") or []))
    pieces.append(_user_prompt_from_val(val))
    return "\n".join(pieces).lower()


def _has_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(cue in text for cue in cues)


def _direction_ok(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    gd = gold.get("direction_tag")
    pd = (pred.get("final_prediction") or {}).get("direction_tag")
    return bool(gd and pd and gd == pd)


def _classify_failure(val: dict[str, Any], pred: dict[str, Any]) -> str:
    gold_f = val.get("family")
    gold_s = val.get("subtype")
    fp = pred.get("final_prediction") or {}
    pred_f = fp.get("family")
    pred_s = fp.get("subtype")
    if not pred.get("parse_ok") or not pred_f:
        return "parse_or_schema_failure"
    if pred_f == gold_f:
        if not _direction_ok(val, pred):
            return "direction_only_error"
        if pred_s != gold_s:
            return "subtype_only_error"
        return "correct"

    if _subtype_equivalent(gold_s, pred_s):
        return "axis_ambiguity_same_subtype"

    text = _trace_text(pred, val)
    if pred_f == "AdverseRisk":
        if gold_f == "PK_Metabolism":
            if _has_any(text, PK_METABOLISM_CUES):
                return "pk_metabolism_mislabeled_as_adverse"
            return "pk_metabolism_to_adverse_no_pk_evidence"
        if gold_f == "PK_Excretion":
            if _has_any(text, PK_EXCRETION_CUES):
                return "pk_excretion_mislabeled_as_adverse"
            return "pk_excretion_to_adverse_no_transport_evidence"
        if gold_f == "PK_Distribution":
            if _has_any(text, PK_DISTRIBUTION_CUES):
                return "pk_distribution_mislabeled_as_adverse"
            return "pk_distribution_to_adverse_no_distribution_evidence"
        if gold_f == "PD_Activity":
            return "pd_activity_mislabeled_as_adverse"
        if gold_f == "Efficacy":
            return "efficacy_mislabeled_as_adverse"
        return "rare_family_mislabeled_as_adverse"

    if gold_f == "AdverseRisk":
        return "adverse_mislabeled_as_non_adverse"

    if gold_f and pred_f and gold_f.startswith("PK_") and pred_f.startswith("PK_"):
        return "pk_family_confusion"

    if gold_f == "Efficacy" and pred_f == "PD_Activity":
        return "efficacy_vs_pd_confusion"
    if gold_f == "PD_Activity" and pred_f == "Efficacy":
        return "pd_vs_efficacy_confusion"
    return "other_family_error"


def _shorten(s: str | None, n: int = 240) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _prediction_summary(pred: dict[str, Any]) -> str:
    fa = ((pred.get("trace") or {}).get("final_answer") or {})
    fp = pred.get("final_prediction") or {}
    return str(fa.get("summary") or fp.get("summary") or "")


def _gold_teacher_summary(val: dict[str, Any]) -> str:
    raw = _assistant_text_from_val(val)
    try:
        obj = json.loads(raw)
        return str((obj.get("final_answer") or {}).get("summary") or "")
    except Exception:
        return ""


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fams = list(FAMILIES)
    tp = Counter()
    fp = Counter()
    fn = Counter()
    total = right = parsed = 0
    for row in rows:
        gold = row["gold_family"]
        pred = row["pred_family"]
        total += 1
        if pred:
            parsed += 1
        if pred == gold:
            right += 1
            tp[gold] += 1
        else:
            if pred in fams:
                fp[pred] += 1
            if gold in fams:
                fn[gold] += 1
    per = {}
    f1s = []
    for fam in fams:
        denom_p = tp[fam] + fp[fam]
        denom_r = tp[fam] + fn[fam]
        prec = tp[fam] / denom_p if denom_p else 0.0
        rec = tp[fam] / denom_r if denom_r else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per[fam] = {
            "tp": tp[fam],
            "fp": fp[fam],
            "fn": fn[fam],
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp[fam] + fn[fam],
        }
    return {
        "n": total,
        "parse_family_present": parsed,
        "accuracy": right / total if total else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_family": per,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_id", "input_order", "cluster", "gold_family", "pred_family",
        "gold_subtype", "pred_subtype", "gold_direction", "pred_direction",
        "confidence", "parse_ok", "subtype_equivalent",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_ab", required=True)
    p.add_argument("--predictions_ba", default=None)
    p.add_argument("--val_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_examples_per_cluster", type=int, default=8)
    p.add_argument("--strict_order", action="store_true",
                   help="Also require AB prediction order to match val order.")
    args = p.parse_args()

    pred_ab_path = Path(args.predictions_ab)
    pred_ba_path = Path(args.predictions_ba) if args.predictions_ba else None
    val_path = Path(args.val_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_rows = _read_jsonl(val_path)
    ab_rows = _read_jsonl(pred_ab_path)
    ba_rows = _read_jsonl(pred_ba_path) if pred_ba_path else []

    integrity = {
        "val": _integrity(val_path, val_rows),
        "predictions_ab": _integrity(pred_ab_path, ab_rows),
    }
    if pred_ba_path:
        integrity["predictions_ba"] = _integrity(pred_ba_path, ba_rows)

    val_by = _index_by_pair(val_rows, "val_file")
    ab_by = _index_by_pair(ab_rows, "predictions_ab")
    if set(val_by) != set(ab_by):
        missing = sorted(set(val_by) - set(ab_by))[:20]
        extra = sorted(set(ab_by) - set(val_by))[:20]
        raise SystemExit(
            "[audit] AB predictions and val_file pair_id sets do not match. "
            f"missing_from_predictions={missing}; extra_in_predictions={extra}. "
            "This usually means the stale Phase-C val file was used."
        )
    if args.strict_order:
        val_order = [r["pair_id"] for r in val_rows]
        ab_order = [r["pair_id"] for r in ab_rows]
        if val_order != ab_order:
            raise SystemExit(
                "[audit] AB predictions and val_file have same pair set but "
                "different order. Refusing because --strict_order was set."
            )
    if ba_rows:
        ba_by = _index_by_pair(ba_rows, "predictions_ba")
        if set(val_by) != set(ba_by):
            missing = sorted(set(val_by) - set(ba_by))[:20]
            extra = sorted(set(ba_by) - set(val_by))[:20]
            raise SystemExit(
                "[audit] BA predictions and val_file pair_id sets do not match. "
                f"missing_from_predictions={missing}; extra_in_predictions={extra}."
            )

    audit_rows: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cluster_counts: Counter = Counter()
    flip_counts: Counter = Counter()
    role_counts_by_cluster: dict[str, Counter] = defaultdict(Counter)
    hint_counts_by_cluster: dict[str, Counter] = defaultdict(Counter)

    for pred in ab_rows:
        pid = pred["pair_id"]
        val = val_by[pid]
        fpred = pred.get("final_prediction") or {}
        cluster = _classify_failure(val, pred)
        pred_family = fpred.get("family")
        gold_family = val.get("family")
        pred_subtype = fpred.get("subtype")
        gold_subtype = val.get("subtype")
        row = {
            "pair_id": pid,
            "input_order": pred.get("input_order", "ab"),
            "cluster": cluster,
            "gold_family": gold_family,
            "pred_family": pred_family,
            "gold_subtype": gold_subtype,
            "pred_subtype": pred_subtype,
            "gold_direction": val.get("direction_tag"),
            "pred_direction": fpred.get("direction_tag"),
            "confidence": fpred.get("confidence"),
            "parse_ok": bool(pred.get("parse_ok")),
            "subtype_equivalent": _subtype_equivalent(gold_subtype, pred_subtype),
        }
        audit_rows.append(row)
        cluster_counts[cluster] += 1
        if cluster != "correct":
            if pred_family != gold_family:
                flip_counts[(gold_family, pred_family)] += 1
            trace = pred.get("trace") or {}
            for step in trace.get("steps") or []:
                role = step.get("role")
                hint = step.get("family_hint")
                if role:
                    role_counts_by_cluster[cluster][role] += 1
                if hint:
                    hint_counts_by_cluster[cluster][hint] += 1
            if len(examples[cluster]) < args.max_examples_per_cluster:
                examples[cluster].append({
                    **row,
                    "student_summary": _shorten(_prediction_summary(pred), 360),
                    "teacher_gold_summary": _shorten(_gold_teacher_summary(val), 360),
                    "first_student_steps": [
                        {
                            "role": s.get("role"),
                            "family_hint": s.get("family_hint"),
                            "claim": _shorten(s.get("claim"), 220),
                            "evidence_ids": s.get("evidence_ids") or [],
                        }
                        for s in ((pred.get("trace") or {}).get("steps") or [])[:4]
                    ],
                })

    metrics = _metrics(audit_rows)
    report = {
        "integrity": integrity,
        "metrics_ab": metrics,
        "cluster_counts": dict(cluster_counts.most_common()),
        "family_flip_counts": {
            f"{g}->{pr}": c for (g, pr), c in flip_counts.most_common()
        },
        "role_counts_by_cluster": {
            k: dict(v.most_common()) for k, v in role_counts_by_cluster.items()
        },
        "hint_counts_by_cluster": {
            k: dict(v.most_common()) for k, v in hint_counts_by_cluster.items()
        },
        "examples": examples,
    }

    (out_dir / "phase4_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n"
    )
    (out_dir / "phase4_error_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    _write_csv(out_dir / "phase4_error_audit.csv", audit_rows)

    with (out_dir / "phase4_error_examples.jsonl").open("w") as f:
        for cluster, items in examples.items():
            for item in items:
                f.write(json.dumps({"cluster": cluster, **item}) + "\n")

    md: list[str] = []
    md.append("# Phase 4.1 Error Audit")
    md.append("")
    md.append("## File Integrity")
    for name, info in integrity.items():
        md.append(
            f"- `{name}`: `{info['path']}` | lines={info['line_count']} | "
            f"unique_pair_ids={info['unique_pair_ids']} | "
            f"sha256=`{info['sha256']}` | first=`{info['first_pair_id']}`"
        )
    md.append("")
    md.append("## AB Metrics")
    md.append(f"- n = {metrics['n']}")
    md.append(
        f"- family-present = {metrics['parse_family_present']}/{metrics['n']}"
    )
    md.append(f"- accuracy = {metrics['accuracy']:.3f}")
    md.append(f"- macro-F1 = {metrics['macro_f1']:.3f}")
    md.append("")
    md.append("### Per Family")
    md.append("| family | support | F1 | precision | recall | TP | FP | FN |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for fam in FAMILIES:
        d = metrics["per_family"][fam]
        md.append(
            f"| {fam} | {d['support']} | {d['f1']:.3f} | "
            f"{d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['tp']} | {d['fp']} | {d['fn']} |"
        )
    md.append("")
    md.append("## Failure Clusters")
    md.append("| cluster | count | share |")
    md.append("|---|---:|---:|")
    n = len(audit_rows)
    for cluster, count in cluster_counts.most_common():
        md.append(f"| `{cluster}` | {count} | {100 * count / n:.1f}% |")
    md.append("")
    md.append("## Family Flips")
    md.append("| gold -> pred | count |")
    md.append("|---|---:|")
    for (gold, pred), count in flip_counts.most_common(25):
        md.append(f"| `{gold}` -> `{pred}` | {count} |")
    md.append("")
    md.append("## Cluster Examples")
    for cluster, items in sorted(examples.items(), key=lambda kv: -cluster_counts[kv[0]]):
        md.append(f"### {cluster} ({cluster_counts[cluster]})")
        for item in items[: args.max_examples_per_cluster]:
            md.append(
                f"- `{item['pair_id']}` gold=`{item['gold_family']}/"
                f"{item['gold_subtype']}` pred=`{item['pred_family']}/"
                f"{item['pred_subtype']}` dir `{item['gold_direction']}` -> "
                f"`{item['pred_direction']}` conf={item['confidence']}"
            )
            if item.get("student_summary"):
                md.append(f"  - student: {item['student_summary']}")
            if item.get("teacher_gold_summary"):
                md.append(f"  - teacher: {item['teacher_gold_summary']}")
        md.append("")
    md.append("## Recommended Phase 4.2 Targets")
    priority = [
        "pk_metabolism_mislabeled_as_adverse",
        "pk_excretion_mislabeled_as_adverse",
        "efficacy_mislabeled_as_adverse",
        "pk_distribution_mislabeled_as_adverse",
        "pd_activity_mislabeled_as_adverse",
        "axis_ambiguity_same_subtype",
    ]
    for cluster in priority:
        if cluster_counts[cluster]:
            md.append(f"- `{cluster}`: {cluster_counts[cluster]} examples")
    (out_dir / "phase4_error_audit.md").write_text("\n".join(md) + "\n")

    print(f"[audit] wrote {out_dir / 'phase4_error_audit.md'}")
    print(f"[audit] wrote {out_dir / 'phase4_error_audit.json'}")
    print(f"[audit] wrote {out_dir / 'phase4_error_audit.csv'}")
    print("[audit] key counts:")
    for cluster, count in cluster_counts.most_common():
        print(f"[audit]   {cluster:45s} {count:5d}")


if __name__ == "__main__":
    main()
