"""evaluation -- OpenDDI neural baselines on the canonical splits.

Runs selected neural DDI architectures from the OpenDDI model zoo
(`external/OpenDDI/openddi/models`) on this repository's canonical
train/val/test manifests and Morgan-fingerprint drug embeddings.
Each model is trained from scratch on the matching train split,
selected on the val split (best macro-F1), and scored on the test
split (or on a stratified subset selected via ``--manifest_jsonl``).

Supported architectures: DeepDDI, DDIMDL, CASTER, KGNN, SumGNN,
DSNDDI, DDKG, MIRACLE, LaGAT, MMDGDTI, ExDDI.

Example
-------
    python -m src.evaluation.openddi_v4_subset \
        --model DeepDDI \
        --split random_full \
        --manifest_jsonl outputs/eval_prompts/random_full_test_5000_stratified.manifest.jsonl \
        --epochs 12 --batch_size 8192

Outputs
-------
* ``outputs/audit/openddi_v4_subset/<model>_<split>_test_5000.json``
  -- metrics summary (best epoch, macro-F1, weighted-F1, accuracy,
  per-class report).
* ``outputs/eval_prompts/openddi_v4_subset/pred_openddi_<model>_<split>_test_5000.jsonl``
  -- per-pair predictions in the standard prediction-record schema
  used by ``src.evaluation.run_full_eval``.

Requires ``external/OpenDDI`` to be present in the repository root
(clone of https://github.com/Mriyazat/OpenDDI or equivalent).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

from src.evaluation.baseline_xgboost import ROOT, DATA, SPLITS_DIR, build_drug_fp_sparse

OPEN_DDI = ROOT / "external" / "OpenDDI" / "openddi"
if str(OPEN_DDI) not in sys.path:
    sys.path.insert(0, str(OPEN_DDI))

OUT_DIR = ROOT / "outputs" / "audit" / "openddi_v4_subset"
PRED_DIR = ROOT / "outputs" / "eval_prompts" / "openddi_v4_subset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MODULES = {
    "DeepDDI": "DeepDDI",
    "DDIMDL": "DDIMDL",
    "CASTER": "CASTER",
    "KGNN": "KGNN",
    "SumGNN": "SumGNN",
    "DSNDDI": "DSNDDI",
    "DDKG": "DDKG",
    "MIRACLE": "MIRACLE",
    "LaGAT": "LaGAT",
    "MMDGDTI": "MMDGDTI",
    "ExDDI": "ExDDI",
}


class PairDataset(Dataset):
    def __init__(self, triples: np.ndarray):
        self.triples = triples.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.triples.shape[0])

    def __getitem__(self, idx: int):
        row = self.triples[idx]
        return int(row[0]), int(row[1]), int(row[2])


def load_manifest_pids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    out = set()
    with path.open() as f:
        for line in f:
            if line.strip():
                out.add(json.loads(line)["pair_id"])
    return out


def load_split(split: str, test_subset: set[str] | None):
    manifest = pq.read_table(SPLITS_DIR / f"manifest_{split}.parquet").to_pylist()
    pairs = pq.read_table(DATA / "pairs.parquet", columns=["pair_id", "a_id", "b_id"]).to_pylist()
    pair_by_pid = {r["pair_id"]: r for r in pairs}

    families = sorted({r["family"] for r in manifest})
    fam2idx = {f: i for i, f in enumerate(families)}
    idx2fam = {i: f for f, i in fam2idx.items()}

    def pids(section: str) -> list[str]:
        rows = [r["pair_id"] for r in manifest if r["split"] == section]
        if section == "test" and test_subset is not None:
            rows = [p for p in rows if p in test_subset]
        return rows

    y_by_pid = {r["pair_id"]: fam2idx[r["family"]] for r in manifest}
    return pids("train"), pids("val"), pids("test"), pair_by_pid, y_by_pid, fam2idx, idx2fam


def make_triples(pids: list[str], pair_by_pid: dict, y_by_pid: dict, id_to_idx: dict, sentinel: int):
    triples = []
    kept_pids = []
    for pid in pids:
        pr = pair_by_pid.get(pid)
        if not pr or pid not in y_by_pid:
            continue
        a = id_to_idx.get(pr["a_id"], sentinel)
        b = id_to_idx.get(pr["b_id"], sentinel)
        y = y_by_pid[pid]
        triples.append((a, b, y))
        kept_pids.append(pid)
    return np.asarray(triples, dtype=np.int64), kept_pids


def build_graph(train_triples: np.ndarray, num_relations: int, edge_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    edges = train_triples
    if 0 < edge_ratio < 1:
        n = max(1, int(round(len(edges) * edge_ratio)))
        edges = edges[rng.choice(len(edges), size=n, replace=False)]

    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    rel = np.concatenate([edges[:, 2], edges[:, 2]])
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_type = torch.tensor(rel % max(num_relations, 1), dtype=torch.long)
    return SimpleNamespace(edge_index=edge_index, edge_type=edge_type)


def move_graph(graph, device):
    graph.x = graph.x.to(device)
    graph.edge_index = graph.edge_index.to(device)
    graph.edge_type = graph.edge_type.to(device)
    return graph


def load_model(name: str, feature: int, hidden1: int, hidden2: int, num_classes: int, dropout: float):
    module = importlib.import_module(f"models.{MODEL_MODULES[name]}")
    cls = getattr(module, name)
    if name == "DDIMDL":
        return cls(features=[feature], hidden1=hidden1, hidden2=hidden2,
                   num_relations=num_classes, num_classes=num_classes, dropout=dropout)
    if name == "PHGLDDI":
        return cls(feature=feature, hidden1=hidden1, hidden2=hidden2,
                   num_relations=num_classes, num_classes=num_classes)
    return cls(feature=feature, hidden1=hidden1, hidden2=hidden2,
               num_relations=num_classes, num_classes=num_classes, dropout=dropout)


def collate(batch):
    a, b, y = zip(*batch)
    return (
        torch.tensor(a, dtype=torch.long),
        torch.tensor(b, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
    )


def forward_model(model, data, batch, device):
    a, b, y = batch
    batch_dev = (a.to(device), b.to(device), y.to(device))
    return model(data, batch_dev), batch_dev[2]


@torch.no_grad()
def evaluate(model, data, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for batch in loader:
        logits, y = forward_model(model, data, batch, device)
        prob = torch.softmax(logits, dim=-1)
        ys.append(y.cpu().numpy())
        ps.append(prob.argmax(dim=-1).cpu().numpy())
        probs.append(prob.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_prob = np.concatenate(probs)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def write_predictions(path: Path, pids: list[str], eval_out: dict, idx2fam: dict):
    with path.open("w") as f:
        for pid, pred, prob in zip(pids, eval_out["y_pred"], eval_out["y_prob"]):
            fam = idx2fam[int(pred)]
            f.write(json.dumps({
                "pair_id": pid,
                "input_order": "ab",
                "model": "openddi_v4_subset",
                "final_prediction": {
                    "family": fam,
                    "subtype": None,
                    "direction_tag": "n/a",
                    "polarity": None,
                    "abstain": False,
                    "confidence": float(prob[int(pred)]),
                    "label_dist": {idx2fam[i]: float(prob[i]) for i in range(len(idx2fam))},
                },
            }) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODEL_MODULES))
    ap.add_argument("--split", required=True, choices=["random_full", "drug_cold", "pair_cold"])
    ap.add_argument("--manifest_jsonl", default=None,
                    help="Optional JSONL manifest of pair_ids to restrict test evaluation to.")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--edge_ratio", type=float, default=0.25,
                    help="Fraction of train edges used in graph models to keep memory bounded.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    subset = load_manifest_pids(Path(args.manifest_jsonl)) if args.manifest_jsonl else None
    train_pids, val_pids, test_pids, pair_by_pid, y_by_pid, fam2idx, idx2fam = load_split(args.split, subset)

    fp_sparse, id_to_idx = build_drug_fp_sparse()
    x_np = fp_sparse.toarray().astype(np.float32)
    x_np = np.vstack([x_np, np.zeros((1, x_np.shape[1]), dtype=np.float32)])
    sentinel = x_np.shape[0] - 1

    train_triples, _ = make_triples(train_pids, pair_by_pid, y_by_pid, id_to_idx, sentinel)
    val_triples, _ = make_triples(val_pids, pair_by_pid, y_by_pid, id_to_idx, sentinel)
    test_triples, kept_test_pids = make_triples(test_pids, pair_by_pid, y_by_pid, id_to_idx, sentinel)

    graph = build_graph(train_triples, len(fam2idx), args.edge_ratio, args.seed)
    graph.x = torch.tensor(x_np, dtype=torch.float32)
    graph = move_graph(graph, device)

    train_loader = DataLoader(PairDataset(train_triples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_triples), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_triples), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = load_model(args.model, x_np.shape[1], args.hidden1, args.hidden2, len(fam2idx), args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = np.bincount(train_triples[:, 2], minlength=len(fam2idx)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

    print(f"[openddi] model={args.model} split={args.split} train={len(train_triples):,} "
          f"val={len(val_triples):,} test={len(test_triples):,} device={device}", flush=True)
    t0 = time.time()
    best = {"val_macro_f1": -1.0, "state": None, "epoch": 0}
    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            logits, y = forward_model(model, graph, batch, device)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate(model, graph, val_loader, device)
        print(f"[openddi] {args.model} {args.split} epoch={ep}/{args.epochs} "
              f"loss={np.mean(losses):.4f} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best["val_macro_f1"]:
            best = {
                "val_macro_f1": val["macro_f1"],
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": ep,
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    test = evaluate(model, graph, test_loader, device)
    report = classification_report(
        test["y_true"], test["y_pred"],
        labels=list(range(len(fam2idx))),
        target_names=[idx2fam[i] for i in range(len(fam2idx))],
        output_dict=True,
        zero_division=0,
    )
    result = {
        "model": args.model,
        "split": args.split,
        "manifest_jsonl": args.manifest_jsonl,
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "test_macro_f1": test["macro_f1"],
        "test_weighted_f1": test["weighted_f1"],
        "test_acc": test["acc"],
        "elapsed_sec": time.time() - t0,
        "per_class": {k: v for k, v in report.items() if isinstance(v, dict)},
    }

    stem = f"{args.model}_{args.split}_test_5000"
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    pred_path = PRED_DIR / f"pred_openddi_{args.model}_{args.split}_test_5000.jsonl"
    write_predictions(pred_path, kept_test_pids, test, idx2fam)
    print(json.dumps({k: result[k] for k in ["model", "split", "best_epoch", "best_val_macro_f1", "test_macro_f1", "test_acc"]}, indent=2), flush=True)
    print(f"[openddi] wrote {OUT_DIR / (stem + '.json')}")
    print(f"[openddi] preds {pred_path}")


if __name__ == "__main__":
    main()
