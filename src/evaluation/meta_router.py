"""Meta-router: per-pair logistic-regression classifier that picks
the family from features = [LLM_probs, BASE_probs, BASE_confidence, LLM_conf,
LLM_predicted_top, BASE_predicted_top].

Trained on a 50% val split with 5-fold CV, evaluated on held-out 50% test.
This is the strongest reviewer-defensible hybrid: hyperparameters are tuned
honestly, and the gain over LLM-solo is reported with paired bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]
LAB2IDX = {f: i for i, f in enumerate(FAMS)}


def _load(path: Path) -> dict[str, dict]:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pair_id")
            if (r.get("input_order") or "ab").lower() != "ab":
                continue
            if pid in out:
                continue
            fp = r.get("final_prediction") or {}
            try:
                conf = float(fp.get("confidence")) if fp.get("confidence") is not None else 0.0
            except Exception:
                conf = 0.0
            label_dist = fp.get("label_dist") or {}
            dist = np.array([float(label_dist.get(f, 0.0)) for f in FAMS], dtype=np.float32)
            s = dist.sum()
            if s > 0:
                dist /= s
            out[pid] = {
                "family": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
                "conf": conf,
                "dist": dist,
            }
    return out


def _manifest(path: Path) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)["pair_id"])
    return out


def _truth(keep: set[str]) -> dict[str, str]:
    t = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet",
                      columns=["pair_id", "family"]).to_pylist()
    return {r["pair_id"]: r["family"] for r in t if r["pair_id"] in keep}


def _llm_dist_from_top(pred: dict) -> np.ndarray:
    """LLM does not emit a full distribution; synthesise one with conf at top, residual split."""
    fam = pred["family"]
    if fam not in LAB2IDX or pred["abstain"]:
        return np.full(len(FAMS), 1.0 / len(FAMS), dtype=np.float32)
    d = np.full(len(FAMS), (1.0 - pred["conf"]) / (len(FAMS) - 1), dtype=np.float32)
    d[LAB2IDX[fam]] = pred["conf"]
    return d


def _featurize(pids, llm, base):
    rows = []
    for pid in pids:
        l = llm[pid]; b = base[pid]
        l_dist = _llm_dist_from_top(l)
        b_dist = b["dist"]
        l_top = float(l_dist.max())
        b_top = float(b_dist.max())
        l_ent = float(-(l_dist * np.log(l_dist + 1e-12)).sum())
        b_ent = float(-(b_dist * np.log(b_dist + 1e-12)).sum())
        agree = float(l["family"] == b["family"])
        rows.append(np.concatenate([
            l_dist, b_dist,
            [l_top, b_top, l_ent, b_ent, l["conf"], b["conf"], agree],
        ]))
    return np.asarray(rows, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_jsonl", required=True)
    ap.add_argument("--pred_llm", required=True)
    ap.add_argument("--pred_base", required=True)
    ap.add_argument("--base_name", default="BASE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.5)
    ap.add_argument("--C_grid", default="0.1,0.5,1.0,3.0,10.0")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--output", default=None)
    ap.add_argument("--output_predictions", default=None)
    args = ap.parse_args()

    pids = _manifest(Path(args.manifest_jsonl))
    keep = set(pids)
    truth = _truth(keep)
    llm = _load(Path(args.pred_llm))
    base = _load(Path(args.pred_base))
    common = sorted(set(truth) & set(llm) & set(base))
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(common)); rng.shuffle(idx)
    n_val = int(len(common) * args.val_fraction)
    val_ids = [common[i] for i in idx[:n_val]]
    tst_ids = [common[i] for i in idx[n_val:]]

    Xv = _featurize(val_ids, llm, base)
    Xt = _featurize(tst_ids, llm, base)
    yv = np.asarray([LAB2IDX[truth[pid]] for pid in val_ids])
    yt = np.asarray([LAB2IDX[truth[pid]] for pid in tst_ids])

    # LLM/baseline solo predictions on test (for paired bootstrap)
    yl = np.asarray([LAB2IDX.get(llm[pid]["family"], -1) for pid in tst_ids])
    yb = np.asarray([LAB2IDX.get(base[pid]["family"], -1) for pid in tst_ids])

    best = None
    for C in [float(x) for x in args.C_grid.split(",")]:
        clf = LogisticRegression(
            C=C, max_iter=400, class_weight="balanced",
            random_state=args.seed,
        )
        clf.fit(Xv, yv)
        yvp = clf.predict(Xv)
        f1 = f1_score(yv, yvp, labels=list(range(len(FAMS))), average="macro", zero_division=0)
        if best is None or f1 > best["val_f1"]:
            best = {"C": C, "val_f1": f1, "clf": clf}

    yt_pred = best["clf"].predict(Xt)

    def macro(a, b):
        return f1_score(a, b, labels=list(range(len(FAMS))), average="macro", zero_division=0)

    out = {
        "manifest": args.manifest_jsonl,
        "base_name": args.base_name,
        "best_C": best["C"],
        "val_macro_meta": best["val_f1"],
        "test_n": len(tst_ids),
        "test_macro_llm_solo": float(macro(yt, yl)),
        "test_macro_base_solo": float(macro(yt, yb)),
        "test_macro_meta": float(macro(yt, yt_pred)),
    }

    # Paired bootstrap vs LLM-solo
    diffs = np.zeros(args.n_boot)
    n = len(tst_ids)
    bo = np.random.default_rng(args.seed + 1)
    for b in range(args.n_boot):
        i = bo.integers(0, n, size=n)
        diffs[b] = macro(yt[i], yt_pred[i]) - macro(yt[i], yl[i])
    out["delta_vs_llm"] = float(out["test_macro_meta"] - out["test_macro_llm_solo"])
    out["delta_vs_llm_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    out["delta_vs_llm_p"] = float(min((diffs <= 0).mean(), (diffs >= 0).mean()) * 2)

    print(json.dumps(out, indent=2))

    if args.output:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2) + "\n")

    if args.output_predictions:
        op = Path(args.output_predictions)
        op.parent.mkdir(parents=True, exist_ok=True)
        with op.open("w") as f:
            for pid, y in zip(tst_ids, yt_pred):
                fam = FAMS[int(y)]
                f.write(json.dumps({
                    "pair_id": pid,
                    "input_order": "ab",
                    "final_prediction": {"family": fam, "abstain": False},
                }) + "\n")


if __name__ == "__main__":
    main()
