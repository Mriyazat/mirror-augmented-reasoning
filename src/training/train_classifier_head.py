"""Phase 3.5 -- Train a linear classifier head on top of the SFT-tuned LoRA.

Why a separate head
-------------------
The Phase 3.3 SFT student passes 4-of-5 hard gates but the family
macro-F1 sits at 0.65 strict / 0.67 lenient.  The dominant failure
mode is an "AdverseRisk attractor": when the rare PK / Efficacy
families are correct, the model often emits AdverseRisk anyway because
generation-time per-token CE concentrates probability mass on the
dominant class regardless of class-balanced sampling.

A linear probe on the LoRA's last-layer hidden state at the
end-of-prompt position bypasses autoregressive generation entirely.
The same representation already encodes the family signal (the SFT
loss made sure of that), but the linear head can be trained with
plain class-weighted softmax CE on the full corpus -- no per-token
generation bias.  Empirically this is worth ~5-10pp macro-F1 on
imbalanced multi-class problems.

What this script does
---------------------
1. Load Qwen2.5-7B + the trained LoRA adapter (eval mode, frozen).
2. Walk the SFT corpus (train + val) and, per record:
     a. apply_chat_template(prompt_msgs, add_generation_prompt=True)
     b. forward the prompt through the LoRA model
     c. take the hidden state of the LAST input position from the
        LAST transformer layer (this is the vector the model would
        use to produce the first generated token, e.g. the assistant's
        opening brace).
3. Cache (hidden, gold_family) tuples to disk so re-training is fast.
4. Fit a class-weighted multinomial logistic regression on train.
5. Evaluate on val (predict + macro-F1 + per-family F1).
6. Save the head weights (numpy .npz with `W`, `b`, `families`,
   `hidden_dim`, `mean`, `std`) for `predict_two_stage.py` to consume.

Output layout
-------------
    <output_dir>/
        head.npz                 -- {W, b, families, hidden_dim, mean, std}
        cached_train.npz         -- (X_train, y_train, pair_ids)
        cached_val.npz           -- (X_val,   y_val,   pair_ids)
        head_metrics.json        -- macroF1, per-family F1, accuracy
        head_metrics.md          -- human-readable summary

Usage
-----
    python -m src.training.train_classifier_head \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter    $DDI_CKPT/student/ddi_v4_sft_reasoning_safe_mirror \
        --train_file outputs/phase_c/teacher_clean.reasoning_safe.train.mirror.jsonl \
        --val_file   outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl \
        --output_dir outputs/student/ddi_v4_sft_reasoning_safe_mirror/head \
        --batch_size 8 \
        --max_length 4096

Inference (separate script, predict_two_stage.py):
    head_logits = X @ W.T + b  ->  softmax over 7 families
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FAMILIES = (
    "AdverseRisk",
    "Efficacy",
    "PD_Activity",
    "PK_Absorption",
    "PK_Distribution",
    "PK_Excretion",
    "PK_Metabolism",
)
FAM_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _build_prompt_text(rec: dict, tokenizer) -> str | None:
    """Return the chat-template-rendered prompt with assistant-generation
    suffix attached.  Skips records that lack a clear assistant turn."""
    msgs = rec.get("messages") or []
    prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
    if not prompt_msgs:
        return None
    return tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True,
    )


def _extract_hidden_states(
    records: list[dict],
    model,
    tokenizer,
    device,
    batch_size: int,
    max_length: int,
    label: str = "split",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run prompt-only forward passes and harvest the last-position
    hidden state from the final transformer layer.

    Returns
    -------
    X       : float32 [N, H]      end-of-prompt last-layer hidden state
    y       : int64   [N]         family index (in FAMILIES order)
    pair_ids: list[str] of length N (for diagnostic dumps)
    """
    import torch

    import sys as _sys
    print(f"[head] {label}: building prompt texts for {len(records)} records...",
          flush=True)
    valid_records = []
    t_build = __import__("time").time()
    for k, rec in enumerate(records):
        fam = rec.get("family")
        if fam not in FAM_TO_IDX:
            continue
        text = _build_prompt_text(rec, tokenizer)
        if text is None:
            continue
        valid_records.append((rec, text))
        if (k + 1) % 5000 == 0:
            elapsed = __import__("time").time() - t_build
            print(f"[head]   {label}: prompt-build {k+1}/{len(records)}  "
                  f"({elapsed:.0f}s elapsed)", flush=True)

    n = len(valid_records)
    print(f"[head] {label}: {n}/{len(records)} records have a usable family + prompt "
          f"(prompt-build done in {__import__('time').time()-t_build:.1f}s)",
          flush=True)

    H = int(model.config.hidden_size)
    X = np.zeros((n, H), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    pair_ids: list[str] = []

    model.eval()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    t_start = __import__("time").time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = valid_records[i : i + batch_size]
            prompts = [t for _, t in batch]
            enc = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]
            attn = enc["attention_mask"]
            last_idx = attn.sum(dim=1) - 1
            arange = torch.arange(last_hidden.size(0), device=device)
            pooled = last_hidden[arange, last_idx, :].float().cpu().numpy()
            for j, (rec, _) in enumerate(batch):
                X[i + j] = pooled[j]
                y[i + j] = FAM_TO_IDX[rec["family"]]
                pair_ids.append(rec.get("pair_id", f"_idx{i+j}"))
            if (i // batch_size) % 25 == 0:
                el = __import__("time").time() - t_start
                rate = (i + len(batch)) / max(el, 1e-6)
                eta = (n - i - len(batch)) / max(rate, 1e-6)
                print(f"[head]   {label} {i+len(batch):>5d}/{n}  "
                      f"{rate:.1f} rec/s  eta={eta/60:.1f} min",
                      flush=True)
    print(f"[head] done {label}: extracted {n} vectors of dim {H}")
    return X, y, pair_ids


def _train_logreg(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weight: str | None = "balanced",
    max_iter: int = 2000,
    C: float = 1.0,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=-1,
    )
    clf.fit(X_tr_s, y_tr)
    pred_val = clf.predict(X_val_s)
    proba_val = clf.predict_proba(X_val_s)
    return clf, scaler, pred_val, proba_val


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray):
    f1s = []
    per_fam = {}
    for k, fam in enumerate(FAMILIES):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        denom_r = tp + fn
        if denom_r == 0:
            per_fam[fam] = {"f1": 0.0, "p": 0.0, "r": 0.0,
                            "tp": tp, "fp": fp, "fn": fn, "n": 0}
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / denom_r
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_fam[fam] = {"f1": f1, "p": prec, "r": rec,
                        "tp": tp, "fp": fp, "fn": fn, "n": denom_r}
    macro = float(sum(f1s) / len(f1s)) if f1s else 0.0
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    return macro, acc, per_fam


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", required=True,
                   help="Path to the trained LoRA adapter (Phase 3.3 output).")
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file",   required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--max_train", type=int, default=-1,
                   help="Cap training records (for quick debug).")
    p.add_argument("--max_val", type=int, default=-1)
    p.add_argument("--cache_only", action="store_true",
                   help="Compute hidden-state caches and exit.")
    p.add_argument("--reuse_cache", action="store_true",
                   help="Skip extraction if cached_{train,val}.npz exist.")
    p.add_argument("--class_weight", default="balanced",
                   choices=["balanced", "none"])
    p.add_argument("--C", type=float, default=1.0,
                   help="L2 regularization (inverse).")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: auto).")
    p.add_argument("--cache_dir", default=None,
                   help="Where to look for / write hidden-state caches. "
                        "Defaults to --output_dir. Pass an explicit path to "
                        "share caches across multiple C / class_weight sweeps "
                        "with different output_dirs.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_train = cache_dir / "cached_train.npz"
    cache_val = cache_dir / "cached_val.npz"

    have_caches = (
        args.reuse_cache and cache_train.exists() and cache_val.exists()
    )

    if have_caches:
        print(f"[head] reusing caches at {out_dir}")
        d_tr = np.load(cache_train, allow_pickle=True)
        X_tr, y_tr, pids_tr = d_tr["X"], d_tr["y"], list(d_tr["pair_ids"])
        d_va = np.load(cache_val, allow_pickle=True)
        X_va, y_va, pids_va = d_va["X"], d_va["y"], list(d_va["pair_ids"])
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        device = (
            args.device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[head] loading base model + adapter on {device}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
            device_map=("auto" if device == "cuda" else None),
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, args.adapter)
        model.eval()

        print("[head] base + adapter loaded; reading SFT corpora...", flush=True)
        train_recs = _load_records(Path(args.train_file))
        val_recs = _load_records(Path(args.val_file))
        if args.max_train > 0:
            train_recs = train_recs[: args.max_train]
        if args.max_val > 0:
            val_recs = val_recs[: args.max_val]
        print(f"[head] train={len(train_recs)}  val={len(val_recs)}", flush=True)

        fam_counts = Counter(r.get("family") for r in train_recs)
        print(f"[head] train family distribution: {dict(fam_counts)}", flush=True)

        X_tr, y_tr, pids_tr = _extract_hidden_states(
            train_recs, model, tokenizer, device,
            args.batch_size, args.max_length, label="train",
        )
        X_va, y_va, pids_va = _extract_hidden_states(
            val_recs, model, tokenizer, device,
            args.batch_size, args.max_length, label="val",
        )

        np.savez_compressed(cache_train, X=X_tr, y=y_tr,
                            pair_ids=np.array(pids_tr, dtype=object))
        np.savez_compressed(cache_val, X=X_va, y=y_va,
                            pair_ids=np.array(pids_va, dtype=object))
        print(f"[head] cached train+val to {out_dir}")

        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.cache_only:
        print("[head] --cache_only set; exiting.")
        return

    cw = None if args.class_weight == "none" else "balanced"
    print(f"[head] fitting logistic regression  C={args.C}  class_weight={cw}  "
          f"X_tr={X_tr.shape}  X_va={X_va.shape}")
    clf, scaler, pred_va, proba_va = _train_logreg(
        X_tr, y_tr, X_va, y_va, class_weight=cw, C=args.C,
    )

    macro, acc, per_fam = _macro_f1(y_va, pred_va)
    macro_tr, acc_tr, per_fam_tr = _macro_f1(y_tr, clf.predict(scaler.transform(X_tr)))

    print(f"\n[head] === results ===")
    print(f"[head] train: macroF1={macro_tr:.3f}  acc={acc_tr:.3f}")
    print(f"[head] val:   macroF1={macro:.3f}  acc={acc:.3f}")
    print(f"[head] val per-family:")
    for f in FAMILIES:
        d = per_fam[f]
        print(f"[head]   {f:18}  F1={d['f1']:.3f}  P={d['p']:.3f}  "
              f"R={d['r']:.3f}  n={d['n']}")

    coef = clf.coef_.astype(np.float32)
    intercept = clf.intercept_.astype(np.float32)

    head_npz = out_dir / "head.npz"
    np.savez_compressed(
        head_npz,
        W=coef,
        b=intercept,
        families=np.array(FAMILIES, dtype=object),
        hidden_dim=int(X_tr.shape[1]),
        mean=scaler.mean_.astype(np.float32),
        std=scaler.scale_.astype(np.float32),
    )
    print(f"[head] saved head -> {head_npz}")

    metrics = {
        "macro_f1_val": macro,
        "accuracy_val": acc,
        "macro_f1_train": macro_tr,
        "accuracy_train": acc_tr,
        "per_family_val": per_fam,
        "per_family_train": per_fam_tr,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_va.shape[0]),
        "hidden_dim": int(X_tr.shape[1]),
        "C": args.C,
        "class_weight": args.class_weight,
        "families": list(FAMILIES),
    }
    (out_dir / "head_metrics.json").write_text(json.dumps(metrics, indent=2))

    md = ["# Phase 3.5 -- classifier head"]
    md.append(f"- adapter: `{args.adapter}`")
    md.append(f"- train: {X_tr.shape[0]} records, val: {X_va.shape[0]} records")
    md.append(f"- hidden dim: {X_tr.shape[1]}")
    md.append(f"- L2 C: {args.C}, class weight: {args.class_weight}")
    md.append("")
    md.append("## Headline")
    md.append("| split | macroF1 | accuracy |")
    md.append("|---|---:|---:|")
    md.append(f"| train | {macro_tr:.3f} | {acc_tr:.3f} |")
    md.append(f"| val   | {macro:.3f} | {acc:.3f} |")
    md.append("")
    md.append("## Validation per-family")
    md.append("| family | F1 | P | R | n |")
    md.append("|---|---:|---:|---:|---:|")
    for f in FAMILIES:
        d = per_fam[f]
        md.append(f"| {f} | {d['f1']:.3f} | {d['p']:.3f} | {d['r']:.3f} | {d['n']} |")
    (out_dir / "head_metrics.md").write_text("\n".join(md) + "\n")

    print(f"[head] wrote metrics -> {out_dir/'head_metrics.json'}")
    print(f"[head] wrote summary -> {out_dir/'head_metrics.md'}")


if __name__ == "__main__":
    main()
