"""Fast baselines for EMNLP-ready comparison.

This script gives us directly comparable baselines on the same splits and
the same 5K stratified manifests used by the student evaluation.

Baselines
---------
1. Majority family: predicts the most frequent training family.
2. Logistic regression: class-weighted multinomial linear model over
   Morgan fingerprints + pair-signature scalars.
3. DeepDDI-style MLP: pair representation
   [fp_a || fp_b || |fp_a - fp_b| || fp_a * fp_b || scalar signatures]
   trained with class-weighted cross entropy.

The MLP is intentionally lightweight. It is not a full OpenDDI port, but it is
the same core neural family as DeepDDI: molecular fingerprints -> pair MLP.
It is a fairer 24-hour baseline than stock OpenDDI on Ryus/twosides because it
uses our train/val/test split and evaluates on the exact same 5K manifest.

Example
-------
    python -m src.evaluation.baseline_fast_suite \\
      --split random_full \\
      --split_section test \\
      --manifest_jsonl outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl \\
      --run_name fast_random_full_5K \\
      --models majority,logreg,mlp \\
      --epochs 8 --batch_size 8192
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score

from src.evaluation.baseline_xgboost import (
    ROOT,
    DATA,
    SPLITS_DIR,
    AUDIT_DIR,
    SCALAR_FEATURES,
    build_drug_fp_sparse,
    build_sparse_matrix,
)


def _load_manifest_pids(path: Path) -> set[str]:
    pids: set[str] = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("pair_id"):
                pids.add(rec["pair_id"])
    return pids


def _load_core_indices():
    print("[fast] loading pairs + signatures + labels ...", flush=True)
    pairs_rows = pq.read_table(
        DATA / "pairs.parquet", columns=["pair_id", "a_id", "b_id"],
    ).to_pylist()
    pair_index_rows = {r["pair_id"]: r for r in pairs_rows}

    sig_rows = pq.read_table(
        DATA / "pair_signatures.parquet",
        columns=["pair_id"] + SCALAR_FEATURES,
    ).to_pylist()
    sig_index = {r["pair_id"]: r for r in sig_rows}

    labels = pq.read_table(
        DATA / "labels_hierarchical.parquet",
        columns=["pair_id", "family"],
    ).to_pylist()
    families = sorted({r["family"] for r in labels})
    fam2idx = {f: i for i, f in enumerate(families)}
    truth_by_pid = {r["pair_id"]: r for r in labels}
    return pair_index_rows, sig_index, fam2idx, truth_by_pid


def _load_split_pids(
    split: str,
    manifest_subset: set[str] | None,
    split_section: str,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    manifest = pq.read_table(SPLITS_DIR / f"manifest_{split}.parquet").to_pylist()
    y_all_raw = {r["pair_id"]: r["family"] for r in manifest}

    train_pids = [r["pair_id"] for r in manifest if r["split"] == "train"]
    val_pids = [r["pair_id"] for r in manifest if r["split"] == "val"]
    test_pids = [r["pair_id"] for r in manifest if r["split"] == "test"]

    if manifest_subset is not None:
        if split_section == "test":
            before = len(test_pids)
            test_pids = [pid for pid in test_pids if pid in manifest_subset]
            print(f"[fast] manifest filter test: {before:,} -> {len(test_pids):,}")
        elif split_section == "val":
            before = len(val_pids)
            val_pids = [pid for pid in val_pids if pid in manifest_subset]
            print(f"[fast] manifest filter val: {before:,} -> {len(val_pids):,}")
        else:
            raise ValueError(f"unsupported split_section={split_section}")

    return train_pids, val_pids, test_pids, y_all_raw


def _assemble_sparse(pids, pair_index_rows, sig_index, fp_matrix, id_to_idx, y_all, fam2idx):
    pairs = [pair_index_rows[pid] for pid in pids]
    X = build_sparse_matrix(pairs, fp_matrix, id_to_idx, sig_index)
    y = np.fromiter((fam2idx[y_all[pid]] for pid in pids), dtype=np.int64, count=len(pids))
    return X, y, pairs


def _score_family_only(name: str, split: str, run_name: str, test_pids: list[str],
                       y_true: np.ndarray, pred: np.ndarray, fam2idx: dict[str, int]) -> dict:
    from src.metrics.ths import score_family_only

    idx2fam = {i: f for f, i in fam2idx.items()}
    pred_fams = [idx2fam[int(i)] for i in pred]
    truth_fams = [idx2fam[int(i)] for i in y_true]
    macro = f1_score(y_true, pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true,
        pred,
        target_names=[f for f, _ in sorted(fam2idx.items(), key=lambda kv: kv[1])],
        output_dict=True,
        zero_division=0,
    )
    ths = score_family_only(pred_fams, truth_fams)
    out = {
        "model": name,
        "split": split,
        "run_name": run_name,
        "test_size": int(len(test_pids)),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
        "ths": {
            "macro_ths": ths["macro_ths"],
            "weighted_ths": ths["weighted_ths"],
            "per_family_ths": ths["per_family_ths"],
            "level_fractions": ths["level_fractions"],
            "scorer": "family_only",
            "max_per_pair": 0.3,
        },
    }
    print(
        f"[fast:{name}] macro_F1={macro:.4f} weighted={weighted:.4f} "
        f"macro_THS={ths['macro_ths']:.4f} (cap 0.3)",
        flush=True,
    )
    return out


def _maybe_write_preds(predictions_out: Path | None, model_name: str,
                       test_pids: list[str], pred: np.ndarray, fam2idx: dict[str, int],
                       proba: np.ndarray | None = None) -> None:
    """Optionally write per-pair predictions JSONL with the same schema as
    `run_full_eval` expects so the downstream evaluator can re-score the
    file unchanged."""
    if predictions_out is None:
        return
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    idx2fam = {i: f for f, i in fam2idx.items()}
    n_classes = len(fam2idx)
    with predictions_out.open("w") as fout:
        for i, pid in enumerate(test_pids):
            pf = idx2fam[int(pred[i])]
            if proba is not None and proba.shape[1] == n_classes:
                probs = proba[i].tolist()
                conf = float(max(probs))
                ldist = {idx2fam[j]: float(probs[j]) for j in range(n_classes)}
            else:
                conf = None
                ldist = None
            fout.write(json.dumps({
                "pair_id":     pid,
                "input_order": "ab",
                "model":       model_name,
                "final_prediction": {
                    "family":        pf,
                    "subtype":       None,
                    "direction_tag": "n/a",
                    "polarity":      None,
                    "abstain":       False,
                    "confidence":    conf,
                    "label_dist":    ldist,
                },
            }) + "\n")
    try:
        rel = predictions_out.relative_to(ROOT)
    except ValueError:
        rel = predictions_out
    print(f"[fast:{model_name}] wrote per-pair predictions to {rel}", flush=True)


def run_majority(split, run_name, train_y, test_y, test_pids, fam2idx,
                 predictions_out: Path | None = None):
    counts = Counter(train_y.tolist())
    maj = counts.most_common(1)[0][0]
    pred = np.full_like(test_y, fill_value=maj)
    out = _score_family_only("majority", split, run_name, test_pids, test_y, pred, fam2idx)
    out["majority_class"] = int(maj)
    _maybe_write_preds(predictions_out, "majority", test_pids, pred, fam2idx)
    return out


def run_logreg(split, run_name, X_tr, y_tr, X_te, y_te, test_pids, fam2idx,
               max_iter: int, predictions_out: Path | None = None):
    print("[fast:logreg] fitting SGD logistic regression ...", flush=True)
    t0 = time.time()
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-3,
        n_jobs=-1,
        random_state=1,
        verbose=0,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    out = _score_family_only("logreg", split, run_name, test_pids, y_te, pred, fam2idx)
    out["train_time_sec"] = time.time() - t0
    out["max_iter"] = max_iter
    # SGDClassifier with log_loss supports predict_proba; use it if available.
    try:
        proba = clf.predict_proba(X_te) if hasattr(clf, "predict_proba") else None
    except Exception:
        proba = None
    _maybe_write_preds(predictions_out, "logreg", test_pids, pred, fam2idx, proba)
    return out


def _build_dense_fp_matrix(fp_sparse):
    # Drug FP matrix is only 19,853 x 2048 (~155 MB as float32 dense).
    return fp_sparse.toarray().astype(np.float32)


def _pair_scalars(pairs, sig_index):
    scalars = np.zeros((len(pairs), len(SCALAR_FEATURES)), dtype=np.float32)
    for i, p in enumerate(pairs):
        sig = sig_index.get(p["pair_id"])
        if not sig:
            continue
        for j, feat in enumerate(SCALAR_FEATURES):
            v = sig.get(feat)
            if v is not None:
                scalars[i, j] = float(v)
    mu = scalars.mean(axis=0, keepdims=True)
    sd = scalars.std(axis=0, keepdims=True) + 1e-6
    return (scalars - mu) / sd


def run_mlp(split, run_name, train_pairs, y_tr, test_pairs, y_te, test_pids,
            fam2idx, fp_matrix, id_to_idx, sig_index, epochs, batch_size, lr, hidden,
            predictions_out: Path | None = None):
    print("[fast:mlp] training DeepDDI-style MLP ...", flush=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp_dense = _build_dense_fp_matrix(fp_matrix)
    sentinel = np.zeros((1, fp_dense.shape[1]), dtype=np.float32)
    fp_dense = np.vstack([fp_dense, sentinel])
    sentinel_idx = fp_dense.shape[0] - 1

    def pair_arrays(pairs):
        a = np.fromiter((id_to_idx.get(p["a_id"], sentinel_idx) for p in pairs),
                        dtype=np.int64, count=len(pairs))
        b = np.fromiter((id_to_idx.get(p["b_id"], sentinel_idx) for p in pairs),
                        dtype=np.int64, count=len(pairs))
        s = _pair_scalars(pairs, sig_index)
        return a, b, s

    tr_a, tr_b, tr_s = pair_arrays(train_pairs)
    te_a, te_b, te_s = pair_arrays(test_pairs)

    class PairMLP(nn.Module):
        def __init__(self, fp_dim, scalar_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4 * fp_dim + scalar_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(hidden // 2, num_classes),
            )

        def forward(self, xa, xb, scalars):
            z = torch.cat([xa, xb, torch.abs(xa - xb), xa * xb, scalars], dim=-1)
            return self.net(z)

    fp_t = torch.tensor(fp_dense, dtype=torch.float32, device=device)
    model = PairMLP(fp_t.shape[1], len(SCALAR_FEATURES), len(fam2idx)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    counts = np.bincount(y_tr, minlength=len(fam2idx)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)

    tr_y = torch.tensor(y_tr, dtype=torch.long, device=device)
    tr_s_t = torch.tensor(tr_s, dtype=torch.float32, device=device)
    n = len(y_tr)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for st in range(0, n, batch_size):
            idx = order[st: st + batch_size]
            a = torch.tensor(tr_a[idx], dtype=torch.long, device=device)
            b = torch.tensor(tr_b[idx], dtype=torch.long, device=device)
            y = tr_y[idx]
            logits = model(fp_t[a], fp_t[b], tr_s_t[idx])
            loss = F.cross_entropy(logits, y, weight=w_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[fast:mlp] epoch {ep}/{epochs} loss={np.mean(losses):.4f}", flush=True)

    model.eval()
    preds = []
    probs_chunks = []
    te_s_t = torch.tensor(te_s, dtype=torch.float32, device=device)
    with torch.no_grad():
        for st in range(0, len(y_te), batch_size):
            sl = slice(st, min(st + batch_size, len(y_te)))
            a = torch.tensor(te_a[sl], dtype=torch.long, device=device)
            b = torch.tensor(te_b[sl], dtype=torch.long, device=device)
            logits = model(fp_t[a], fp_t[b], te_s_t[sl])
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            probs_chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    proba = np.concatenate(probs_chunks, axis=0)
    out = _score_family_only("deepddi_mlp", split, run_name, test_pids, y_te, pred, fam2idx)
    out["train_time_sec"] = time.time() - t0
    out["epochs"] = epochs
    out["batch_size"] = batch_size
    out["hidden"] = hidden
    out["lr"] = lr
    _maybe_write_preds(predictions_out, "deepddi_mlp", test_pids, pred, fam2idx, proba)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["random_full", "drug_cold", "pair_cold"])
    ap.add_argument("--split_section", default="test", choices=["test", "val"])
    ap.add_argument("--manifest_jsonl", default=None)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--models", default="majority,logreg,mlp",
                    help="Comma-separated: majority,logreg,mlp")
    ap.add_argument("--logreg_max_iter", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--predictions_out_dir", default=None,
                    help="If set, write per-pair predictions JSONL for every "
                         "model that ran into this directory as "
                         "preds_<model>_<run_name>.jsonl. "
                         "Compatible with run_full_eval / leakage_probe / "
                         "ensemble_eval.")
    args = ap.parse_args()

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pair_index_rows, sig_index, fam2idx, _truth = _load_core_indices()
    subset = _load_manifest_pids(Path(args.manifest_jsonl)) if args.manifest_jsonl else None
    train_pids, _val_pids, test_pids, y_all_raw = _load_split_pids(
        args.split, subset, args.split_section,
    )

    print("[fast] building Morgan FP sparse matrix ...", flush=True)
    fp_matrix, id_to_idx = build_drug_fp_sparse()
    print(f"[fast] fp matrix shape={fp_matrix.shape} nnz={fp_matrix.nnz:,}", flush=True)

    X_tr, y_tr, train_pairs = _assemble_sparse(
        train_pids, pair_index_rows, sig_index, fp_matrix, id_to_idx, y_all_raw, fam2idx,
    )
    X_te, y_te, test_pairs = _assemble_sparse(
        test_pids, pair_index_rows, sig_index, fp_matrix, id_to_idx, y_all_raw, fam2idx,
    )
    print(f"[fast] train={X_tr.shape} test={X_te.shape}", flush=True)

    preds_dir = Path(args.predictions_out_dir) if args.predictions_out_dir else None
    if preds_dir is not None:
        preds_dir.mkdir(parents=True, exist_ok=True)

    def _pp(model_label: str) -> Path | None:
        if preds_dir is None:
            return None
        return preds_dir / f"preds_{model_label}_{args.run_name}.jsonl"

    wanted = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    results = {}
    if "majority" in wanted:
        results["majority"] = run_majority(
            args.split, args.run_name, y_tr, y_te, test_pids, fam2idx,
            predictions_out=_pp("majority"),
        )
    if "logreg" in wanted:
        results["logreg"] = run_logreg(
            args.split, args.run_name, X_tr, y_tr, X_te, y_te, test_pids, fam2idx,
            args.logreg_max_iter,
            predictions_out=_pp("logreg"),
        )
    if "mlp" in wanted:
        results["deepddi_mlp"] = run_mlp(
            args.split, args.run_name, train_pairs, y_tr, test_pairs, y_te, test_pids,
            fam2idx, fp_matrix, id_to_idx, sig_index, args.epochs, args.batch_size,
            args.lr, args.hidden,
            predictions_out=_pp("deepddi_mlp"),
        )

    out_path = AUDIT_DIR / f"fast_baselines_{args.run_name}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[fast] wrote {out_path.relative_to(ROOT)}")

    md = [
        f"# Fast baselines: {args.run_name}",
        "",
        "| model | macro-F1 | weighted-F1 | macro-THS |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        md.append(
            f"| {name} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
            f"{r['ths']['macro_ths']:.4f} |"
        )
    md_path = AUDIT_DIR / f"fast_baselines_{args.run_name}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[fast] wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
