"""End-to-end smoke test for the evaluation harness.

Synthesizes a tiny predictions JSONL + labels parquet and invokes the
harness via subprocess.  Confirms the headline metrics file is produced
and that per-family breakdowns populate correctly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def _mk_trace(family: str, subtype: str, direction: str, evidence_ids: list[str] | None = None,
              abstain: bool = False, confidence: float | None = 0.8) -> dict:
    """Minimal V4 trace + final_prediction pair."""
    eids = evidence_ids or []
    trace = {
        "steps": [{
            "step_id":      1,
            "role":         "mechanism_of_action",
            "claim":        f"Drug X inhibits pathway causing {family}.",
            "evidence_ids": eids,
        }, {
            "step_id":      2,
            "role":         "evidence_resolution",
            "claim":        "Confirmed from DrugBank.",
            "evidence_ids": eids,
        }],
        "final_answer": {
            "family":        family,
            "subtype":       subtype,
            "direction_tag": direction,
            "polarity":      "antagonize",
            "abstain":       abstain,
        },
    }
    fp = {
        "family":        family,
        "subtype":       subtype,
        "direction_tag": direction,
        "polarity":      "antagonize",
        "abstain":       abstain,
        "confidence":    confidence,
        "label_dist":    {family: 0.8, "NONE": 0.2},
    }
    return {"trace": trace, "final_prediction": fp}


def test_eval_harness_end_to_end(tmp_path: Path):
    # 10 pairs, each with (ab, ba) predictions -> 20 records
    pred_path = tmp_path / "preds.jsonl"
    lbl_path  = tmp_path / "labels.parquet"
    out_dir   = tmp_path / "eval_out"

    preds: list[dict] = []
    rows:  list[dict] = []
    for i in range(10):
        pair_id = f"P{i:03d}"
        family  = "PK_Metabolism" if i < 6 else "PD_Receptor"
        subtype = "inhibits_CYP3A4" if family == "PK_Metabolism" else "antagonist"
        direction = "a_to_b" if i % 2 == 0 else "bidirectional"
        eids = [f"DB_cyp3a4_{i}", f"enzyme_CYP3A4"]
        # AB -> correct; BA -> flip direction if directional
        direction_ba = {"a_to_b": "b_to_a", "b_to_a": "a_to_b",
                        "bidirectional": "bidirectional"}[direction]

        ab = _mk_trace(family, subtype, direction, eids)
        ba = _mk_trace(family, subtype, direction_ba, eids)

        preds.append({"pair_id": pair_id, "input_order": "ab",
                      "context_ids": eids, **ab})
        preds.append({"pair_id": pair_id, "input_order": "ba",
                      "context_ids": eids, **ba})

        rows.append({
            "pair_id":              pair_id,
            "family":               family,
            "subtype":              subtype,
            "bidirectional":        direction == "bidirectional",
            "subject_drugbank_id":  "DA" if direction == "a_to_b" else None,
            "object_drugbank_id":   "DB" if direction == "a_to_b" else None,
            "polarity":             "antagonize",
            "a_id":                 "DA", "b_id": "DB",
            "severity":             "Moderate",
        })

    with open(pred_path, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    pq.write_table(pa.Table.from_pylist(rows), lbl_path)

    cmd = [
        sys.executable, "-m", "src.evaluation.run_full_eval",
        "--predictions", str(pred_path),
        "--labels",      str(lbl_path),
        "--run_name",    "smoke",
        "--output_dir",  str(out_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"

    metrics_json = out_dir / "metrics.json"
    metrics_md   = out_dir / "metrics.md"
    assert metrics_json.exists(), "metrics.json not produced"
    assert metrics_md.exists(),   "metrics.md not produced"

    full = json.loads(metrics_json.read_text())
    hl = full["headline"]
    # Everything we predicted is "correct" against labels
    assert hl["family_accuracy"] > 0.9, f"acc {hl['family_accuracy']}"
    # MFS should be high because every step has resolved evidence
    assert hl["mfs"] > 0.5, f"mfs {hl['mfs']}"
    # MPS: since we flipped directionals correctly, MPS should be high
    assert hl["mps"] > 0.9, f"mps {hl['mps']}"
    # CSA: since families are stable across orderings, CSA should be high
    assert hl["csa"] > 0.9, f"csa {hl['csa']}"

    # per-family exists and covers both families
    fams = set(full["per_family"].keys())
    assert {"PK_Metabolism", "PD_Receptor"} <= fams, f"missing families: {fams}"

    print("EVAL HARNESS E2E OK")
    print(f"  mfs = {hl['mfs']:.3f}   mps = {hl['mps']:.3f}   csa = {hl['csa']:.3f}")
    print(f"  rpc = {hl['rpc']:.3f}   slfs = {hl['slfs']:.3f}   hr = {hl['hr']:.3f}")
    print(f"  ths = {hl['ths']:.3f}   acc = {hl['family_accuracy']:.3f}")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_eval_harness_end_to_end(Path(td))
