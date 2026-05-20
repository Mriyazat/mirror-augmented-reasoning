"""Prepare Phase C student-training files from the final clean teacher release.

This script is intentionally CPU-only and cluster-portable. It validates the
canonical SFT corpus and mirror-preference file, creates deterministic
train/validation splits, and writes small smoke-test subsets for GPU preflight.

Example:
    python -m src.data.prepare_phase_c \
        --teacher_file /path/to/teacher_clean.reasoning_safe.jsonl \
        --pref_file /path/to/mirror_preferences.reasoning_safe.jsonl \
        --out_dir outputs/phase_c
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUTS_DIR = Path(
    os.environ.get("DDI_OUTPUTS", "outputs")
)
DEFAULT_TEACHER_DIR = DEFAULT_OUTPUTS_DIR / "teacher_final_clean"
REQUIRED_SFT_KEYS = {
    "pair_id",
    "family",
    "subtype",
    "direction_tag",
    "polarity",
    "messages",
    "tier",
    "sample_weight",
}
REQUIRED_FINAL_KEYS = {
    "family",
    "subtype",
    "direction_tag",
    "polarity",
    "confidence",
    "abstain",
    "summary",
}
ALLOWED_TIERS = {"full_correct", "family_correct", "abstention", "near_miss"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def _assistant_payload(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if len(assistant_msgs) != 1:
        raise ValueError("record must contain exactly one assistant message")
    content = assistant_msgs[0].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("assistant content must be a non-empty string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant content is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("assistant content JSON must be an object")
    return payload


def _validate_sft(records: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    tiers: Counter[str] = Counter()
    families: Counter[str] = Counter()
    final_families: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    label_mismatches: dict[str, Counter[str]] = defaultdict(Counter)
    abstain_count = 0

    for idx, record in enumerate(records):
        pair_id = record.get("pair_id")
        missing = sorted(REQUIRED_SFT_KEYS - set(record))
        if missing:
            errors.append(f"record {idx}: missing keys {missing}")
            continue
        if not isinstance(pair_id, str) or not pair_id:
            errors.append(f"record {idx}: bad pair_id")
            continue
        if pair_id in seen:
            errors.append(f"record {idx}: duplicate pair_id {pair_id}")
            continue
        seen.add(pair_id)

        tier = record.get("tier")
        family = record.get("family")
        if tier not in ALLOWED_TIERS:
            errors.append(f"{pair_id}: bad tier {tier!r}")
        if not isinstance(family, str) or not family:
            errors.append(f"{pair_id}: bad family {family!r}")

        try:
            payload = _assistant_payload(record)
        except ValueError as exc:
            errors.append(f"{pair_id}: {exc}")
            continue

        steps = payload.get("steps")
        final = payload.get("final_answer")
        if not isinstance(steps, list) or not 3 <= len(steps) <= 8:
            errors.append(f"{pair_id}: steps must be a list of length 3..8")
        if not isinstance(final, dict):
            errors.append(f"{pair_id}: final_answer must be an object")
            continue
        missing_final = sorted(REQUIRED_FINAL_KEYS - set(final))
        if missing_final:
            errors.append(f"{pair_id}: final_answer missing {missing_final}")
        if bool(final.get("abstain")):
            abstain_count += 1
        for key in ("family", "subtype", "direction_tag", "polarity"):
            if final.get(key) != record.get(key):
                label_mismatches[str(tier)][key] += 1
                # Top-level labels are gold/metadata; assistant labels are the
                # training target. Mismatches are expected for family_correct,
                # near_miss, and abstention tiers, and bidirectional gold may
                # permit a more specific committed direction.
                if tier == "full_correct" and key != "direction_tag":
                    warnings.append(
                        f"{pair_id}: full_correct final {key} differs from metadata"
                    )
        tiers[str(tier)] += 1
        families[str(family)] += 1
        final_families[str(final.get("family"))] += 1
        strata[f"{tier}|{family}"] += 1

    return {
        "records": len(records),
        "unique_pair_ids": len(seen),
        "errors": errors,
        "warnings": warnings,
        "tiers": dict(tiers),
        "families": dict(families),
        "final_families": dict(final_families),
        "strata": dict(strata),
        "label_mismatches_by_tier": {
            tier: dict(counter) for tier, counter in label_mismatches.items()
        },
        "abstain_count": abstain_count,
    }


def _validate_preferences(records: list[dict[str, Any]], known_pair_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    mirror_types: Counter[str] = Counter()
    missing_pair_ids = 0

    for idx, record in enumerate(records):
        for key in ("pair_id", "prompt", "chosen", "rejected", "mirror_type"):
            if key not in record:
                errors.append(f"preference {idx}: missing {key}")
        pair_id = record.get("pair_id")
        if pair_id not in known_pair_ids:
            missing_pair_ids += 1
        if not isinstance(record.get("prompt"), list):
            errors.append(f"preference {idx}: prompt must be a message list")
        if not isinstance(record.get("chosen"), str) or not record.get("chosen"):
            errors.append(f"preference {idx}: chosen must be a non-empty string")
        if not isinstance(record.get("rejected"), str) or not record.get("rejected"):
            errors.append(f"preference {idx}: rejected must be a non-empty string")
        mirror_types[str(record.get("mirror_type", "unknown"))] += 1

    if missing_pair_ids:
        errors.append(f"{missing_pair_ids} preference records reference pair_ids outside SFT corpus")

    return {
        "records": len(records),
        "errors": errors,
        "mirror_types": dict(mirror_types),
    }


def _stable_shuffle(records: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    rng.shuffle(records)


def _split_sft(
    records: list[dict[str, Any]],
    seed: int,
    val_frac: float,
    val_size: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[f"{record.get('tier')}|{record.get('family')}"].append(record)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for stratum, group in sorted(by_stratum.items()):
        _stable_shuffle(group, seed + int(hashlib.sha256(stratum.encode()).hexdigest()[:8], 16))
        if len(group) == 1:
            n_val = 0
        else:
            n_val = max(1, round(len(group) * val_frac))
            n_val = min(n_val, len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    _stable_shuffle(train, seed)
    _stable_shuffle(val, seed + 1)

    if val_size is not None and val_size > 0 and len(val) > val_size:
        overflow = val[val_size:]
        val = val[:val_size]
        train.extend(overflow)
        _stable_shuffle(train, seed + 2)

    return train, val


def _split_preferences(
    prefs: list[dict[str, Any]],
    val_pair_ids: set[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [p for p in prefs if p.get("pair_id") not in val_pair_ids]
    val = [p for p in prefs if p.get("pair_id") in val_pair_ids]
    _stable_shuffle(train, seed)
    _stable_shuffle(val, seed + 1)
    return train, val


def _hist(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(r.get(key)) for r in records))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase C Preparation Report",
        "",
        "## Inputs",
        "",
        f"- SFT corpus: `{report['inputs']['teacher_file']}`",
        f"- Preference corpus: `{report['inputs']['pref_file']}`",
        "",
        "## SFT Split",
        "",
        f"- Train records: {report['sft_split']['train_records']:,}",
        f"- Val records: {report['sft_split']['val_records']:,}",
        f"- Seed: {report['args']['seed']}",
        "",
        "## Preference Split",
        "",
        f"- Train preferences: {report['pref_split']['train_records']:,}",
        f"- Val preferences: {report['pref_split']['val_records']:,}",
        "",
        "## Validation",
        "",
        f"- SFT validation errors: {len(report['validation']['sft_errors'])}",
        f"- SFT validation warnings: {len(report['validation']['sft_warnings'])}",
        f"- Preference validation errors: {len(report['validation']['pref_errors'])}",
        "",
        "## Output Files",
        "",
    ]
    for name, value in report["outputs"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "```bash",
            "python -m src.data.prepare_phase_c --teacher_file $TEACHER_FILE --pref_file $PREF_FILE --out_dir outputs/sft_corpus",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher_file",
        type=Path,
        default=DEFAULT_TEACHER_DIR / "teacher_clean.reasoning_safe.jsonl",
    )
    parser.add_argument(
        "--pref_file",
        type=Path,
        default=DEFAULT_TEACHER_DIR / "mirror_preferences.reasoning_safe.jsonl",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/phase_c"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.08)
    parser.add_argument("--val_size", type=int, default=2000)
    parser.add_argument("--allow_validation_errors", action="store_true")
    parser.add_argument(
        "--commits_only",
        action="store_true",
        help="Keep only tiers whose trace has the CORRECT family in its "
             "final_answer.  By construction:"
             "  full_correct  = predicted family AND subtype match gold "
             "                  (right family in trace)"
             "  family_correct= predicted family matches, subtype doesn't "
             "                  (right family, wrong subtype)"
             "  near_miss     = ALL teachers got the family wrong "
             "                  (WRONG family in trace) -- DROPPED"
             "  abstention    = trace abstains -- DROPPED"
             "Including near_miss teaches the student to predict the wrong "
             "family on those pairs, which directly hurts macro-F1.  "
             "Abstention is held for separate abstention-head training.",
    )
    parser.add_argument(
        "--include_tiers",
        default="full_correct,family_correct,abstention,near_miss",
        help="Comma-separated whitelist of tiers to keep. Overrides "
             "--commits_only when both are set. Use this to ablate "
             "individual tiers (e.g. 'full_correct' for a pure-gold ablation).",
    )
    parser.add_argument(
        "--low_consensus_penalty",
        type=float,
        default=0.7,
        help="Multiplier applied to sample_weight for records whose "
             "consensus.k_family <= --low_consensus_threshold.  k_family=1 "
             "means only ONE of the three teachers committed to a family, "
             "so the chosen trace is single-teacher signal and ~23%% of "
             "such traces have a wrong family vs the multi-teacher "
             "consensus.  Default 0.7 (30%% downweight); set to 1.0 to "
             "disable.",
    )
    parser.add_argument(
        "--low_consensus_threshold",
        type=int,
        default=1,
        help="k_family threshold below which to apply --low_consensus_penalty. "
             "Default 1 = penalise single-voter records.",
    )
    args = parser.parse_args()

    if not args.teacher_file.exists():
        raise SystemExit(f"teacher_file not found: {args.teacher_file}")
    if not args.pref_file.exists():
        raise SystemExit(f"pref_file not found: {args.pref_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sft_records = _load_jsonl(args.teacher_file)
    pref_records = _load_jsonl(args.pref_file)

    # Optional tier filter -- runs BEFORE validation/splitting so the report
    # numbers reflect what actually goes into Phase C training.
    explicit_tiers = (
        args.include_tiers != "full_correct,family_correct,abstention,near_miss"
    )
    if args.commits_only or explicit_tiers:
        if args.commits_only and not explicit_tiers:
            # NOTE: this set INTENTIONALLY excludes near_miss.  Earlier
            # versions kept near_miss under --commits_only, but near_miss
            # records have predicted_family != gold_family by definition
            # (the merge_consensus selector only assigns near_miss when
            # ALL 3 teachers got the family wrong).  Including them is
            # equivalent to teaching the student wrong family on ~3% of
            # the SFT corpus; that directly hurts macro-F1.
            keep = {"full_correct", "family_correct"}
        else:
            keep = {t.strip() for t in args.include_tiers.split(",") if t.strip()}
        bad = keep - ALLOWED_TIERS
        if bad:
            raise SystemExit(
                f"--include_tiers refers to unknown tier(s): {sorted(bad)}. "
                f"Allowed: {sorted(ALLOWED_TIERS)}"
            )
        before = len(sft_records)
        sft_records = [r for r in sft_records if r.get("tier") in keep]
        dropped = before - len(sft_records)
        # Drop preferences that no longer have a corresponding SFT record.
        keep_pids = {r.get("pair_id") for r in sft_records}
        before_pref = len(pref_records)
        pref_records = [
            p for p in pref_records if p.get("pair_id") in keep_pids
        ]
        dropped_pref = before_pref - len(pref_records)
        print(
            f"[prepare_phase_c] tier filter active: keep={sorted(keep)} "
            f"sft kept={len(sft_records):,} dropped={dropped:,} "
            f"pref kept={len(pref_records):,} dropped={dropped_pref:,}"
        )

    # Low-consensus penalty.  Apply BEFORE validation/splitting so that
    # the report's avg sample_weight reflects what the student actually
    # sees.  We only modify sample_weight; nothing else changes.
    if args.low_consensus_penalty < 1.0:
        n_penalised = 0
        weight_before = 0.0
        weight_after = 0.0
        for r in sft_records:
            kfam = (r.get("consensus") or {}).get("k_family")
            try:
                kfam = int(kfam) if kfam is not None else None
            except (TypeError, ValueError):
                kfam = None
            w = float(r.get("sample_weight") or 0.0)
            weight_before += w
            if kfam is not None and kfam <= args.low_consensus_threshold:
                w *= args.low_consensus_penalty
                r["sample_weight"] = round(w, 4)
                n_penalised += 1
            weight_after += w
        print(
            f"[prepare_phase_c] low-consensus penalty: "
            f"k_family<={args.low_consensus_threshold} × "
            f"{args.low_consensus_penalty:.2f}  "
            f"affected={n_penalised:,}/{len(sft_records):,}  "
            f"total_weight {weight_before:.0f} → {weight_after:.0f} "
            f"({100 * weight_after / max(1e-9, weight_before):.1f}%)"
        )

    sft_validation = _validate_sft(sft_records)
    pref_validation = _validate_preferences(
        pref_records, {str(r["pair_id"]) for r in sft_records if "pair_id" in r}
    )

    sft_errors = sft_validation["errors"]
    pref_errors = pref_validation["errors"]
    if (sft_errors or pref_errors) and not args.allow_validation_errors:
        error_preview = "\n".join((sft_errors + pref_errors)[:20])
        raise SystemExit(
            "validation failed; use --allow_validation_errors only for debugging\n"
            + error_preview
        )

    train_sft, val_sft = _split_sft(
        sft_records,
        seed=args.seed,
        val_frac=args.val_frac,
        val_size=args.val_size,
    )
    val_pair_ids = {str(r["pair_id"]) for r in val_sft}
    train_pref, val_pref = _split_preferences(pref_records, val_pair_ids, args.seed)

    outputs = {
        "sft_train": args.out_dir / "reasoning_safe.train.jsonl",
        "sft_val": args.out_dir / "reasoning_safe.val.jsonl",
        "pref_train": args.out_dir / "mirror_preferences.reasoning_safe.train.jsonl",
        "pref_val": args.out_dir / "mirror_preferences.reasoning_safe.val.jsonl",
        "report_json": args.out_dir / "sft_corpus_prep_report.json",
        "report_md": args.out_dir / "sft_corpus_prep_report.md",
    }

    _write_jsonl(outputs["sft_train"], train_sft)
    _write_jsonl(outputs["sft_val"], val_sft)
    _write_jsonl(outputs["pref_train"], train_pref)
    _write_jsonl(outputs["pref_val"], val_pref)

    report = {
        "args": {
            "seed": args.seed,
            "val_frac": args.val_frac,
            "val_size": args.val_size,
        },
        "inputs": {
            "teacher_file": str(args.teacher_file),
            "pref_file": str(args.pref_file),
            "teacher_sha256": _sha256(args.teacher_file),
            "pref_sha256": _sha256(args.pref_file),
        },
        "validation": {
            "sft_errors": sft_errors,
            "sft_warnings": sft_validation["warnings"],
            "pref_errors": pref_errors,
            "sft_tiers": sft_validation["tiers"],
            "sft_families": sft_validation["families"],
            "sft_final_families": sft_validation["final_families"],
            "sft_label_mismatches_by_tier": sft_validation["label_mismatches_by_tier"],
            "pref_mirror_types": pref_validation["mirror_types"],
        },
        "sft_split": {
            "train_records": len(train_sft),
            "val_records": len(val_sft),
            "train_tiers": _hist(train_sft, "tier"),
            "val_tiers": _hist(val_sft, "tier"),
            "train_families": _hist(train_sft, "family"),
            "val_families": _hist(val_sft, "family"),
        },
        "pref_split": {
            "train_records": len(train_pref),
            "val_records": len(val_pref),
            "train_mirror_types": _hist(train_pref, "mirror_type"),
            "val_mirror_types": _hist(val_pref, "mirror_type"),
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
    }

    outputs["report_json"].write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_report(outputs["report_md"], report)

    print(f"[prepare_phase_c] wrote {args.out_dir}")
    print(f"  SFT train/val:  {len(train_sft):,} / {len(val_sft):,}")
    print(f"  Pref train/val: {len(train_pref):,} / {len(val_pref):,}")
    print(f"  Report: {outputs['report_md']}")


if __name__ == "__main__":
    main()
