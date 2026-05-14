"""A8.2 — XGBoost family-classification baseline on the three canonical splits.

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
"""
from __future__ import annotations

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


def run_split(split_name: str, fp_matrix, id_to_idx, pair_index_rows, sig_index, fam2idx,
              truth_by_pid: dict) -> dict:
    from src.metrics.ths import score_family_only
    print(f"\n[A8-xgb] ==== {split_name} ====", flush=True)
    manifest = pq.read_table(SPLITS_DIR / f"manifest_{split_name}.parquet").to_pylist()
    y_all = {r["pair_id"]: fam2idx[r["family"]] for r in manifest}

    train_pids = [r["pair_id"] for r in manifest if r["split"] == "train"]
    val_pids   = [r["pair_id"] for r in manifest if r["split"] == "val"]
    test_pids  = [r["pair_id"] for r in manifest if r["split"] == "test"]
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
    (AUDIT_DIR / f"a8_xgb_{split_name}.json").write_text(json.dumps(out, indent=2))
    print(f"[A8-xgb]   macro_F1 = {macro_f1:.4f}, weighted = {weighted_f1:.4f}, "
          f"macro_THS(family-only) = {ths_stats['macro_ths']:.4f}  (cap 0.3)", flush=True)
    return out


def main():
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

    results = {}
    for split in ("random_full", "drug_cold", "pair_cold"):
        results[split] = run_split(split, fp_matrix, id_to_idx, pair_index_rows,
                                    sig_index, fam2idx, truth_by_pid)

    # Report markdown
    md = ["# A8.2 — XGBoost baseline on V4 splits\n",
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
