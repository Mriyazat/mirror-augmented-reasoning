"""Augment a predictions JSONL with gold labels joined from
`data_processed/labels_hierarchical.parquet`, and (optionally) attach
per-pair retrieval `context_ids` needed by MFS / HR / RIS.

`src/inference/abstention.py fit` expects `gold_family` on every
prediction record so it can calibrate the conformal threshold on a
*labeled* calibration set. The output of `src/inference/predict.py`
doesn't carry gold labels (predict.py is label-blind by design).
This script does the join.

What it adds to each record (top-level fields):
    gold_family
    gold_subtype
    gold_direction_tag     ('a_to_b' | 'b_to_a' | 'bidirectional', derived)
    gold_polarity
    gold_bidirectional
    gold_subject_drugbank_id
    gold_object_drugbank_id
    gold_a_id
    gold_b_id
    context_ids            (if --build_contexts, sorted list[str])

Deriving gold_direction_tag from the parquet:
    bidirectional == True                         -> 'bidirectional'
    subject_drugbank_id == a_id                   -> 'a_to_b'
    subject_drugbank_id == b_id                   -> 'b_to_a'
    anything else (missing subject)               -> 'bidirectional'
The generating student sees (a_id, b_id) as "Drug A, Drug B" in the
order their pair_id is emitted (ab). The mirror pass flips that to
(b_id, a_id); for a `--input_order=ba` record we therefore invert the
direction_tag so it matches the direction relative to the *presented*
order.

`context_ids` notes:
    - Built via `src.teacher.context_builder.ContextBuilder.build(pair_id)`,
      which is the same pool the teacher used to ground claims.
    - Expensive to construct the first time (~20s to load all parquets)
      but each pair is O(1) after that. Results are cached per-pair_id,
      so AB and BA records share a single build.
    - Required for MFS / HR / RIS to score any trace > 0. Without it,
      every evidence_id fails to resolve and MFS pegs at 0.
    - Skipped when --no_build_contexts is set (e.g. to feed an abstention
      fit that does not need contexts).

Usage
-----
    python -m src.inference.augment_predictions \
        --predictions outputs/phase_c/predictions_val.jsonl \
        --labels      data_processed/labels_hierarchical.parquet \
        --output      outputs/phase_c/predictions_val.with_gold.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pyarrow.parquet as pq
    _HAVE_PQ = True
except Exception as _pq_err:
    _HAVE_PQ = False
    _PQ_ERR = _pq_err


def _load_labels(path: str | Path) -> dict[str, dict]:
    if not _HAVE_PQ:
        raise RuntimeError(
            f"pyarrow required to read {path}. "
            f"Install via `pip install pyarrow`. Underlying: {_PQ_ERR!r}"
        )
    tbl = pq.read_table(path).to_pylist()
    return {r["pair_id"]: r for r in tbl}


def _derive_gold_direction(lab: dict, input_order: str) -> str:
    """Gold direction relative to the *presented* order.

    If input_order == 'ab' the presented pair is (a_id, b_id). If
    input_order == 'ba' the presented pair is (b_id, a_id) and a
    gold 'a_to_b' in the canonical layout becomes 'b_to_a' in the
    presented layout.
    """
    if lab.get("bidirectional"):
        return "bidirectional"
    subj = lab.get("subject_drugbank_id")
    a_id = lab.get("a_id")
    b_id = lab.get("b_id")
    if subj is None or subj not in (a_id, b_id):
        return "bidirectional"
    canonical = "a_to_b" if subj == a_id else "b_to_a"
    if input_order == "ba":
        return {"a_to_b": "b_to_a", "b_to_a": "a_to_b"}.get(canonical, canonical)
    return canonical


def augment(pred_path: Path, labels_path: Path, out_path: Path,
            build_contexts: bool = True,
            overwrite_context_ids: bool = False) -> dict:
    """Join gold labels (and optionally context_ids) onto a predictions JSONL.

    Args:
        pred_path: input predictions JSONL.
        labels_path: parquet with gold labels (see module docstring).
        out_path: output JSONL.
        build_contexts: if True, attach `context_ids` per record from
            `ContextBuilder`. Required for MFS / HR / RIS to score > 0.
        overwrite_context_ids: if True, replace existing non-empty
            `context_ids`. Default False preserves any ids already present.
    """
    labels = _load_labels(labels_path)

    cb = None
    ctx_cache: dict[str, list[str]] = {}
    n_ctx_built = 0
    n_ctx_failed = 0
    if build_contexts:
        from src.teacher.context_builder import ContextBuilder
        cb = ContextBuilder()

    n = 0
    n_joined = 0
    n_missing = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            pid = r.get("pair_id")

            if cb is not None and pid:
                if overwrite_context_ids or not r.get("context_ids"):
                    if pid not in ctx_cache:
                        try:
                            ctx_cache[pid] = sorted(cb.build(pid).context_ids())
                            n_ctx_built += 1
                        except Exception:
                            ctx_cache[pid] = []
                            n_ctx_failed += 1
                    r["context_ids"] = ctx_cache[pid]

            lab = labels.get(pid)
            if lab is None:
                n_missing += 1
                fout.write(json.dumps(r) + "\n")
                continue
            n_joined += 1
            input_order = r.get("input_order", "ab")
            r["gold_family"] = lab.get("family")
            r["gold_subtype"] = lab.get("subtype")
            r["gold_polarity"] = lab.get("polarity")
            r["gold_bidirectional"] = bool(lab.get("bidirectional", False))
            r["gold_subject_drugbank_id"] = lab.get("subject_drugbank_id")
            r["gold_object_drugbank_id"] = lab.get("object_drugbank_id")
            r["gold_a_id"] = lab.get("a_id")
            r["gold_b_id"] = lab.get("b_id")
            r["gold_direction_tag"] = _derive_gold_direction(lab, input_order)
            fout.write(json.dumps(r) + "\n")
    return {
        "n": n,
        "n_joined": n_joined,
        "n_missing_from_labels": n_missing,
        "n_context_built": n_ctx_built,
        "n_context_failed": n_ctx_failed,
        "build_contexts": build_contexts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--labels", required=True,
                   help="Path to data_processed/labels_hierarchical.parquet.")
    p.add_argument("--output", required=True)
    p.add_argument("--no_build_contexts", dest="build_contexts",
                   action="store_false",
                   help="Skip ContextBuilder (faster; but MFS/HR/RIS will "
                        "score 0 for every trace). Default: build contexts.")
    p.add_argument("--overwrite_context_ids", action="store_true",
                   help="Replace existing non-empty context_ids. Default off.")
    p.set_defaults(build_contexts=True)
    args = p.parse_args()

    if not _HAVE_PQ:
        raise SystemExit(
            f"pyarrow required. torch/transformers NOT required for this step. "
            f"Install pyarrow: pip install pyarrow. {_PQ_ERR!r}"
        )

    stats = augment(
        Path(args.predictions),
        Path(args.labels),
        Path(args.output),
        build_contexts=args.build_contexts,
        overwrite_context_ids=args.overwrite_context_ids,
    )
    print(f"[augment] wrote {args.output}")
    print(f"[augment] n={stats['n']}  joined={stats['n_joined']}  "
          f"missing_from_labels={stats['n_missing_from_labels']}")
    if stats["build_contexts"]:
        print(f"[augment] context_ids: built for {stats['n_context_built']} "
              f"unique pair_ids  failed={stats['n_context_failed']}")
    else:
        print("[augment] context_ids: skipped (--no_build_contexts)")


if __name__ == "__main__":
    main()
