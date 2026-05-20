"""XGBoost baseline — XGBoost family-classification baseline on the three canonical splits.

Purpose (sanity gate): validate that our drug_cold / pair_cold splits really
lose information relative to random_full — i.e. no leakage. We expect a
sizeable macro-F1 drop from random_full → drug_cold → pair_cold. If not, the
splits are leaky.

Features per pair (concatenated symmetrically, then ordered by drug id):
  - Morgan FP (radius 2, 2048 bits) for drug_A  →  2048 bits
  - Morgan FP for drug_B                        →  2048 bits
  - Scalar signatures (pair_signatures.parquet):
      pathway_jaccard, protein_jaccard, smiles_tanimoto, atc_prefix_depth,
      n_pathways_a, n_pathways_b, n_proteins_a, n_proteins_b  →  8 floats
  → total 4104 features per pair, stored as scipy CSR.

Target: hierarchical family (8 classes).

For each split:
  1. Build sparse feature matrices for train/val/test.
  2. Train XGBoost with tree_method='hist', early stopping on val.
  3. Compute macro-F1 + per-family F1 on test.
  4. Write metrics to outputs/audit/a8_xgb_<split>.json
Final audit report: outputs/audit/a8_xgboost_report.md

Optional CLI for fair LLM↔baseline comparison
---------------------------------------------
By default the script runs all three canonical splits on their full
test partitions. To compare against the LLM run on a stratified subset
(the same 5K manifest the student saw), pass the matching args:

    python -m src.evaluation.baseline_xgboost \
        --split random_full \
        --split_section test \
        --manifest_jsonl outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl \
        --run_name xgboost_random_full_test_5000_stratified

`--manifest_jsonl` filters the test partition to exactly the pair_ids
in the manifest (one JSON record per line, each with `pair_id`).
Training is **always** done on the full train partition — only test
evaluation is filtered. `--split` restricts the run to that one split
instead of all three.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import classification_report, f1_score
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import xgboost as xgb

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
SPLITS_DIR = DATA / "splits"
AUDIT_DIR = ROOT / "outputs" / "audit"
AUDIT_MD = AUDIT_DIR / "a8_xgboost_report.md"

FP_BITS = 2048
FP_RADIUS = 2
SCALAR_FEATURES = ["pathway_jaccard", "protein_jaccard", "smiles_tanimoto",
                   "atc_prefix_depth", "n_pathways_a", "n_pathways_b",
                   "n_proteins_a", "n_proteins_b"]

# XGBoost hyperparams (fast, reasonable for sanity baseline).
# num_class is set dynamically in main() from labels_hierarchical.parquet.
XGB_PARAMS = {
    "objective": "multi:softprob",
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "nthread": 10,
    "verbosity": 1,
}
NUM_BOOST_ROUND = 300
EARLY_STOP = 20


def build_drug_fp_sparse() -> tuple[csr_matrix, dict[str, int]]:
    """Build a sparse drug × FP_BITS matrix and an id-to-row index.

    Drugs without a valid SMILES get an all-zero row.
    """
    from rdkit import DataStructs
    drugs = pq.read_table(DATA / "drugs.parquet",
                          columns=["drugbank_id", "smiles"]).to_pylist()
    gen = AllChem.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    n = len(drugs)
    # Collect row/col indices for on-bits
    rows: list[int] = []
    cols: list[int] = []
    id_to_idx: dict[str, int] = {}
    for i, d in enumerate(drugs):
        id_to_idx[d["drugbank_id"]] = i
        smi = d.get("smiles")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        arr = np.zeros(FP_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        on_bits = np.where(arr)[0]
        rows.extend([i] * len(on_bits))
        cols.extend(on_bits.tolist())
    data = np.ones(len(rows), dtype=np.float32)
    fp_matrix = csr_matrix((data, (rows, cols)), shape=(n, FP_BITS), dtype=np.float32)
    return fp_matrix, id_to_idx


def build_sparse_matrix(pairs: list[dict], fp_matrix: csr_matrix,
                        id_to_idx: dict[str, int],
                        sig_index: dict[str, dict]) -> csr_matrix:
    """Efficient sparse feature construction using row indexing and hstack."""
    from scipy.sparse import vstack as sp_vstack
    n = len(pairs)
    # Sentinel zero row for drugs missing from the fp table
    zero_row = csr_matrix((1, fp_matrix.shape[1]), dtype=np.float32)
    fp_ext = sp_vstack([fp_matrix, zero_row], format="csr")
    sentinel = fp_ext.shape[0] - 1

    a_idx = np.fromiter((id_to_idx.get(p["a_id"], sentinel) for p in pairs),
                        dtype=np.int64, count=n)
    b_idx = np.fromiter((id_to_idx.get(p["b_id"], sentinel) for p in pairs),
                        dtype=np.int64, count=n)
    fp_a = fp_ext[a_idx]
    fp_b = fp_ext[b_idx]

    # Scalar feature matrix: dense, small
    scalars = np.zeros((n, len(SCALAR_FEATURES)), dtype=np.float32)
    for i, p in enumerate(pairs):
        sig = sig_index.get(p["pair_id"])
        if not sig:
            continue
        for j, feat in enumerate(SCALAR_FEATURES):
            v = sig.get(feat)
            if v is not None:
                scalars[i, j] = float(v)
    scalars_sp = csr_matrix(scalars)

    return hstack([fp_a, fp_b, scalars_sp], format="csr")


def _load_manifest_pids(path: Path) -> set[str]:
    """Read a JSONL manifest and return the set of pair_ids it references."""
    pids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("pair_id")
            if pid:
                pids.add(pid)
    return pids


def _maybe_write_xgb_preds(predictions_out: Path | None, test_pids: list[str],
                            pred: np.ndarray, prob: np.ndarray, fam2idx: dict):
    """Emit per-pair predictions in run_full_eval-compatible JSONL."""
    if predictions_out is None:
        return
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    idx2fam = {i: f for f, i in fam2idx.items()}
    with predictions_out.open("w") as fout:
        for i, pid in enumerate(test_pids):
            fam = idx2fam[int(pred[i])]
            conf = float(prob[i, int(pred[i])])
            rec = {
                "pair_id":     pid,
                "input_order": "ab",
                "final_prediction": {
                    "family":        fam,
                    "subtype":       None,
                    "direction_tag": None,
                    "abstain":       False,
                    "confidence":    conf,
                    "label_dist":    {idx2fam[j]: float(prob[i, j])
                                       for j in range(prob.shape[1])},
                    "aggregator":    "xgboost",
                },
            }
            fout.write(json.dumps(rec) + "\n")
    print(f"[A8-xgb]   per-pair predictions -> {predictions_out}")


def run_split(split_name: str, fp_matrix, id_to_idx, pair_index_rows, sig_index, fam2idx,
              truth_by_pid: dict,
              manifest_subset: set[str] | None = None,
              section_filter: str | None = None,
              run_name: str | None = None,
              predictions_out: Path | None = None) -> dict:
    from src.metrics.ths import score_family_only
    print(f"\n[A8-xgb] ==== {split_name}{' / ' + run_name if run_name else ''} ====",
          flush=True)
    manifest = pq.read_table(SPLITS_DIR / f"manifest_{split_name}.parquet").to_pylist()
    y_all = {r["pair_id"]: fam2idx[r["family"]] for r in manifest}

    train_pids = [r["pair_id"] for r in manifest if r["split"] == "train"]
    val_pids   = [r["pair_id"] for r in manifest if r["split"] == "val"]
    test_pids  = [r["pair_id"] for r in manifest if r["split"] == "test"]

    if manifest_subset is not None:
        # Filter the relevant section to exactly the manifest's pair_ids.
        # Training stays on the full train partition (so the model is the
        # same one a 5K stratified eval should be compared against).
        sec = (section_filter or "test").lower()
        if sec == "test":
            kept = [pid for pid in test_pids if pid in manifest_subset]
            if not kept:
                raise ValueError(
                    f"manifest had {len(manifest_subset)} pair_ids but none "
                    f"were in {split_name}.test ({len(test_pids)} pairs)."
                )
            print(f"[A8-xgb]   manifest filter on test: "
                  f"{len(test_pids):,} -> {len(kept):,}", flush=True)
            test_pids = kept
        elif sec == "val":
            kept = [pid for pid in val_pids if pid in manifest_subset]
            print(f"[A8-xgb]   manifest filter on val: "
                  f"{len(val_pids):,} -> {len(kept):,}", flush=True)
            val_pids = kept
        else:
            print(f"[A8-xgb]   WARN: unknown --split_section {section_filter!r}, "
                  f"ignoring manifest filter.", flush=True)

    print(f"[A8-xgb]   sizes: train={len(train_pids):,}  val={len(val_pids):,}  "
          f"test={len(test_pids):,}", flush=True)

    def assemble(pids):
        pairs = [pair_index_rows[pid] for pid in pids]
        X = build_sparse_matrix(pairs, fp_matrix, id_to_idx, sig_index)
        y = np.fromiter((y_all[pid] for pid in pids), dtype=np.int32, count=len(pids))
        return X, y

    t0 = time.time()
    X_tr, y_tr = assemble(train_pids)
    X_va, y_va = assemble(val_pids)
    X_te, y_te = assemble(test_pids)
    print(f"[A8-xgb]   feature build: {time.time()-t0:.0f}s; "
          f"train nnz={X_tr.nnz:,}, test nnz={X_te.nnz:,}", flush=True)

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_va, label=y_va)
    dtest  = xgb.DMatrix(X_te, label=y_te)

    t0 = time.time()
    bst = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dval, "val")],
                    early_stopping_rounds=EARLY_STOP,
                    verbose_eval=25)
    train_time = time.time() - t0
    print(f"[A8-xgb]   trained in {train_time:.0f}s, best_ntree={bst.best_iteration}", flush=True)

    # Evaluate on test
    prob = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    pred = np.argmax(prob, axis=1)

    _maybe_write_xgb_preds(predictions_out, test_pids, pred, prob, fam2idx)

    macro_f1 = f1_score(y_te, pred, average="macro")
    micro_f1 = f1_score(y_te, pred, average="micro")
    weighted_f1 = f1_score(y_te, pred, average="weighted")
    report = classification_report(
        y_te, pred,
        target_names=[fam for fam, _ in sorted(fam2idx.items(), key=lambda x: x[1])],
        output_dict=True, zero_division=0)

    # ── THS (family-only scorer — XGBoost doesn't predict subtype/direction) ─
    # Max achievable per-pair THS is W_FAMILY_ONLY = 0.3, since the baseline
    # is not in the running for the subtype or direction levels.  Reported so
    # the eventual student's THS has a baseline number to beat.
    idx2fam = {i: f for f, i in fam2idx.items()}
    pred_fams = [idx2fam[int(i)] for i in pred]
    truth_fams = [truth_by_pid[pid]["family"] for pid in test_pids]
    ths_stats = score_family_only(pred_fams, truth_fams)

    out = {
        "split": split_name,
        "run_name": run_name,
        "manifest_filtered": manifest_subset is not None,
        "section_filter": section_filter,
        "train_size": int(len(train_pids)),
        "val_size": int(len(val_pids)),
        "test_size": int(len(test_pids)),
        "train_time_sec": train_time,
        "best_ntree": int(bst.best_iteration),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
        "ths": {
            "macro_ths": ths_stats["macro_ths"],
            "weighted_ths": ths_stats["weighted_ths"],
            "per_family_ths": ths_stats["per_family_ths"],
            "level_fractions": ths_stats["level_fractions"],
            "scorer": "family_only",
            "max_per_pair": 0.3,
            "note": ths_stats.get("note", ""),
        },
    }
    audit_name = (
        f"a8_xgb_{run_name}.json" if run_name
        else f"a8_xgb_{split_name}.json"
    )
    (AUDIT_DIR / audit_name).write_text(json.dumps(out, indent=2))
    print(f"[A8-xgb]   macro_F1 = {macro_f1:.4f}, weighted = {weighted_f1:.4f}, "
          f"macro_THS(family-only) = {ths_stats['macro_ths']:.4f}  (cap 0.3)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None,
                    choices=[None, "random_full", "drug_cold", "pair_cold"],
                    help="Restrict run to this single split. Default: run all three.")
    ap.add_argument("--split_section", default="test",
                    choices=["test", "val"],
                    help="Which section the manifest_jsonl filter applies to.")
    ap.add_argument("--manifest_jsonl", default=None,
                    help="JSONL manifest (one pair_id per line in a 'pair_id' "
                         "field). When set, evaluation is restricted to those "
                         "pair_ids; training stays on the full train partition.")
    ap.add_argument("--run_name", default=None,
                    help="Name for the audit output (a8_xgb_<run_name>.json). "
                         "Defaults to the split name.")
    ap.add_argument("--predictions_out", default=None,
                    help="Optional path to write per-pair JSONL predictions "
                         "(run_full_eval-compatible schema). When set with "
                         "--split, the file holds only that split's test "
                         "predictions; without --split, the path is treated "
                         "as a template with {split} placeholder.")
    args = ap.parse_args()

    print("[A8-xgb] loading pairs + signatures ...", flush=True)
    pairs_rows = pq.read_table(DATA / "pairs.parquet",
                               columns=["pair_id", "a_id", "b_id"]).to_pylist()
    pair_index_rows = {r["pair_id"]: r for r in pairs_rows}

    sig_rows = pq.read_table(DATA / "pair_signatures.parquet",
                             columns=["pair_id"] + SCALAR_FEATURES).to_pylist()
    sig_index = {r["pair_id"]: r for r in sig_rows}

    # Family label space + full truth index (for THS scoring)
    labels = pq.read_table(DATA / "labels_hierarchical.parquet",
                           columns=["pair_id", "family", "subtype", "polarity",
                                    "subject_drugbank_id", "object_drugbank_id",
                                    "bidirectional", "a_id", "b_id"]).to_pylist()
    families = sorted({r["family"] for r in labels})
    fam2idx = {f: i for i, f in enumerate(families)}
    truth_by_pid = {r["pair_id"]: r for r in labels}
    print(f"[A8-xgb] families ({len(fam2idx)}): {families}")

    print("[A8-xgb] building Morgan FP sparse matrix for all drugs ...", flush=True)
    t0 = time.time()
    fp_matrix, id_to_idx = build_drug_fp_sparse()
    print(f"[A8-xgb]   fp matrix: shape={fp_matrix.shape}, nnz={fp_matrix.nnz:,}  "
          f"in {time.time()-t0:.0f}s")

    # Pin num_class now that we know the family count
    XGB_PARAMS["num_class"] = len(fam2idx)

    manifest_subset: set[str] | None = None
    if args.manifest_jsonl:
        manifest_subset = _load_manifest_pids(Path(args.manifest_jsonl))
        print(f"[A8-xgb] manifest filter loaded: "
              f"{len(manifest_subset):,} pair_ids from "
              f"{args.manifest_jsonl}", flush=True)

    splits_to_run = [args.split] if args.split else [
        "random_full", "drug_cold", "pair_cold",
    ]
    results = {}
    for split in splits_to_run:
        pred_path: Path | None = None
        if args.predictions_out:
            tmpl = args.predictions_out
            pred_path = Path(tmpl.format(split=split) if "{split}" in tmpl else tmpl)
        results[split] = run_split(
            split, fp_matrix, id_to_idx, pair_index_rows,
            sig_index, fam2idx, truth_by_pid,
            manifest_subset=manifest_subset if split == args.split else None,
            section_filter=args.split_section,
            run_name=args.run_name if split == args.split else None,
            predictions_out=pred_path,
        )

    if len(results) == 1:
        # Single-split mode (often used for the LLM↔baseline 5K compare).
        # The legacy report below is only meaningful when all three splits
        # ran; skip it here.
        only = next(iter(results.values()))
        print(f"[A8-xgb] single-split summary  split={args.split}  "
              f"run_name={args.run_name}  "
              f"macro_F1={only['macro_f1']:.4f}  "
              f"weighted_F1={only['weighted_f1']:.4f}  "
              f"macro_THS={only['ths']['macro_ths']:.4f}  "
              f"(test_size={only['test_size']:,})")
        return

    # Report markdown
    md = ["# XGBoost baseline — XGBoost baseline on the canonical splits\n",
          f"- Features: 2×2048 Morgan FP + 8 signature scalars = {2*FP_BITS + len(SCALAR_FEATURES)} dims",
          f"- Model: xgb {xgb.__version__}, tree_method=hist, max_depth={XGB_PARAMS['max_depth']}, "
          f"eta={XGB_PARAMS['eta']}, early_stop={EARLY_STOP}",
          f"- Classes: {len(fam2idx)} (`{'`, `'.join(families)}`)",
          "",
          "## Headline results",
          "",
          "| Split | train | val | test | macro-F1 | weighted-F1 | macro-THS (fam-only, cap 0.3) | time (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for split, r in results.items():
        md.append(f"| `{split}` | {r['train_size']:,} | {r['val_size']:,} | "
                  f"{r['test_size']:,} | **{r['macro_f1']:.4f}** | "
                  f"{r['weighted_f1']:.4f} | {r['ths']['macro_ths']:.4f} | "
                  f"{r['train_time_sec']:.0f} |")
    md.append("")
    md.append("## Per-family F1 on test")
    md.append("")
    md.append("| Family | random_full | drug_cold | pair_cold |")
    md.append("|---|---:|---:|---:|")
    for fam in families:
        row = [f"`{fam}`"]
        for split in ("random_full", "drug_cold", "pair_cold"):
            pc = results[split]["per_class"].get(fam, {})
            row.append(f"{pc.get('f1-score', 0):.3f}")
        md.append("| " + " | ".join(row) + " |")
    md += [
        "",
        "## Gates",
        f"- random_full macro-F1 > pair_cold macro-F1: "
        f"**{'PASS' if results['random_full']['macro_f1'] > results['pair_cold']['macro_f1'] else 'FAIL'}** "
        f"(expected large gap since pair_cold removes all drug overlap)",
        f"- drug_cold macro-F1 ≤ random_full macro-F1: "
        f"**{'PASS' if results['drug_cold']['macro_f1'] <= results['random_full']['macro_f1'] else 'FAIL'}**",
        f"- Split gap (random_full − pair_cold): "
        f"{results['random_full']['macro_f1'] - results['pair_cold']['macro_f1']:.4f}"
        f"  ← larger = more inductive difficulty (evidence of leakage-free splits)",
    ]

    AUDIT_MD.write_text("\n".join(md) + "\n")
    print(f"[A8-xgb] wrote {AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
