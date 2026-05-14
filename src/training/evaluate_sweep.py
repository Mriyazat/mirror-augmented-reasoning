"""Backfill validation metrics on already-trained Phase C sweep checkpoints.

Why this exists
---------------
The first cloud LR sweep shipped with `eval_steps=200` while `max_steps=169`,
so Trainer never actually evaluated. The four sweep adapters under
`$DDI_CKPT/student/sweeps/lr*_r*` are still usable -- we just need to
retroactively score them on the held-out val split. This is cheaper than
redoing the sweep.

What it does
------------
For each sweep run directory:
  1.  Load the base model (defaults to Qwen/Qwen2.5-7B-Instruct).
  2.  Attach the adapter at run_dir (or run_dir/checkpoint-XXX, newest).
  3.  Build a SftJsonlDataset over --val_file using the SAME collator as
      training (so weighting + masking match).
  4.  Call Trainer.evaluate -> grab eval_loss and eval_assistant_token_acc.
  5.  Append a dict to trainer_state.json's log_history so downstream
      summarize_sweep can rank as if eval had fired.

Usage
-----
    python -m src.training.evaluate_sweep \\
        --sweep_dir ddi_checkpoints_v4/student/sweeps \\
        --val_file  outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl

    # Evaluate one run only (e.g. for debugging):
    python -m src.training.evaluate_sweep \\
        --sweep_dir ddi_checkpoints_v4/student/sweeps/lr3e-4_r64 \\
        --val_file  outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl \\
        --single

Memory
------
Runs on a single GPU (adapter add + eval only). batch_per_gpu default = 1
to keep it simple. Increase via --batch_per_gpu if you have headroom.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

try:
    import torch
    _HAVE_TORCH = True
except Exception as _torch_err:
    _HAVE_TORCH = False
    _TORCH_ERR = _torch_err

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    )
    from peft import PeftModel
    _HAVE_HF = True
except Exception as _hf_err:
    _HAVE_HF = False
    _HF_ERR = _hf_err

from src.training.sft_train import (
    SftJsonlDataset, _collate, _compute_eval_metrics,
    _preprocess_logits_for_metrics, DdiSftTrainer, _IGNORE_INDEX,
)


def _latest_checkpoint(run_dir: Path) -> Path:
    """Pick the newest 'checkpoint-*' inside run_dir. Fall back to run_dir."""
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return run_dir
    ckpts.sort(key=lambda p: int(p.name.split("-")[-1])
               if p.name.split("-")[-1].isdigit() else 0)
    return ckpts[-1]


def _run_dirs(sweep_root: Path, single: bool) -> list[Path]:
    if single:
        return [sweep_root]
    return [d for d in sorted(sweep_root.iterdir())
            if d.is_dir() and any(d.glob("checkpoint-*"))]


def _build_model(base_name: str, adapter_dir: Path, dtype_str: str):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=dtype, trust_remote_code=True,
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, trust_remote_code=True,
        )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model


def _patch_trainer_state(run_dir: Path, step: int, eval_loss: float,
                        token_acc: float | None, eval_tokens: int | None):
    """Append an eval entry to trainer_state.json in the run dir and its latest
    checkpoint, so summarize_sweep ranks by eval_loss."""
    for target in [_latest_checkpoint(run_dir), run_dir]:
        ts_path = target / "trainer_state.json"
        if not ts_path.exists():
            continue
        state = json.loads(ts_path.read_text())
        state.setdefault("log_history", [])
        entry = {
            "eval_loss": float(eval_loss),
            "step": int(step),
            "epoch": state.get("epoch"),
            "backfilled": True,
        }
        if token_acc is not None:
            entry["eval_assistant_token_acc"] = float(token_acc)
        if eval_tokens is not None:
            entry["eval_assistant_tokens"] = int(eval_tokens)
        state["log_history"].append(entry)
        ts_path.write_text(json.dumps(state, indent=2))


def _evaluate_one(run_dir: Path, val_file: Path, base_model: str,
                  batch_per_gpu: int, max_length: int, dtype: str) -> dict:
    adapter_dir = _latest_checkpoint(run_dir)
    print(f"[evaluate_sweep] {run_dir.name}: adapter={adapter_dir.name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(base_model, adapter_dir, dtype)

    eval_ds = SftJsonlDataset(val_file, tokenizer, max_length=max_length)

    def _collator(batch: list[dict]) -> dict:
        return _collate(batch, tokenizer, max_length)

    targs = TrainingArguments(
        output_dir=str(run_dir / "_eval_tmp"),
        per_device_eval_batch_size=batch_per_gpu,
        do_train=False,
        do_eval=True,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=(dtype == "bfloat16"),
        fp16=(dtype == "float16"),
        dataloader_num_workers=2,
        seed=42,
    )
    trainer_kwargs = dict(
        model=model,
        args=targs,
        eval_dataset=eval_ds,
        data_collator=_collator,
        lambda_sym=0.0,
        lambda_faith=0.0,
        compute_metrics=_compute_eval_metrics,
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
    )
    init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DdiSftTrainer(**trainer_kwargs)

    metrics = trainer.evaluate()
    out = {
        "run": run_dir.name,
        "adapter": adapter_dir.name,
        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
        "eval_assistant_token_acc":
            float(metrics.get("eval_assistant_token_acc", float("nan")))
            if "eval_assistant_token_acc" in metrics else None,
        "eval_assistant_tokens":
            int(metrics.get("eval_assistant_tokens", 0))
            if "eval_assistant_tokens" in metrics else None,
        "n_val_records": len(eval_ds),
    }
    # patch trainer_state so summarize_sweep works
    state = json.loads(
        (adapter_dir / "trainer_state.json").read_text()
        if (adapter_dir / "trainer_state.json").exists()
        else '{"global_step": 0, "epoch": null, "log_history": []}'
    )
    _patch_trainer_state(
        run_dir,
        step=state.get("global_step", 0),
        eval_loss=out["eval_loss"],
        token_acc=out.get("eval_assistant_token_acc"),
        eval_tokens=out.get("eval_assistant_tokens"),
    )
    print(f"[evaluate_sweep]   eval_loss={out['eval_loss']:.4f}  "
          f"token_acc={out.get('eval_assistant_token_acc')}")

    # free GPU memory before the next run
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True,
                   help="Parent sweep dir, or a single run dir if --single.")
    p.add_argument("--val_file", required=True,
                   help="Path to val JSONL (same schema as SftJsonlDataset).")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--batch_per_gpu", type=int, default=1)
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--single", action="store_true",
                   help="Treat --sweep_dir as a single run dir.")
    p.add_argument("--out", default=None, help="Write summary JSON here.")
    args = p.parse_args()

    if not (_HAVE_TORCH and _HAVE_HF):
        raise SystemExit(
            "torch / transformers / peft must be importable. "
            f"torch ok={_HAVE_TORCH}, hf ok={_HAVE_HF}"
        )

    sweep_root = Path(args.sweep_dir)
    run_dirs = _run_dirs(sweep_root, args.single)
    print(f"[evaluate_sweep] runs: {[d.name for d in run_dirs]}")

    results = []
    for d in run_dirs:
        try:
            res = _evaluate_one(
                d, Path(args.val_file), args.base_model,
                args.batch_per_gpu, args.max_length, args.torch_dtype,
            )
            results.append(res)
        except Exception as e:
            print(f"[evaluate_sweep] FAILED on {d.name}: {e!r}")
            results.append({"run": d.name, "error": repr(e)})

    out_path = Path(args.out) if args.out else sweep_root / "backfill_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[evaluate_sweep] wrote {out_path}")

    # Human-readable ranked table
    ranked = sorted(
        [r for r in results if "eval_loss" in r],
        key=lambda r: r["eval_loss"],
    )
    print("\n[evaluate_sweep] ranked by eval_loss (lower better):")
    for r in ranked:
        acc = r.get("eval_assistant_token_acc")
        acc_s = f"{acc:.4f}" if isinstance(acc, float) and not np.isnan(acc) else "—"
        print(f"  {r['run']:<16} eval_loss={r['eval_loss']:.4f}  token_acc={acc_s}")


if __name__ == "__main__":
    main()
