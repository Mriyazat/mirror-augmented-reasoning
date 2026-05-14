"""Post-hoc summary of a Phase C LR sweep.

Reads every `$SWEEP_DIR/*/checkpoint-*/trainer_state.json` and emits a
ranked markdown + JSON report. Never re-runs evaluation; this is the
cheap "what happened?" tool.

If a run has no recorded eval (e.g. eval_steps > max_steps bug, as in the
original sweep), the row still renders with the best training loss and
the max grad_norm, and the script marks `needs_backfill=true` in the
JSON so callers can run `src.training.evaluate_sweep` on it.

Usage
-----
    python -m src.training.summarize_sweep \\
        --sweep_dir ddi_checkpoints_v4/student/sweeps \\
        --out       ddi_checkpoints_v4/student/sweeps/sweep_summary.md \\
        --out_json  ddi_checkpoints_v4/student/sweeps/sweep_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_trainer_state(run_dir: Path) -> dict | None:
    """Prefer the last checkpoint's trainer_state; fall back to run root."""
    ckpts = sorted(run_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1])
                   if p.name.split("-")[-1].isdigit() else 0)
    for ck in reversed(ckpts):
        ts = ck / "trainer_state.json"
        if ts.exists():
            return json.loads(ts.read_text())
    ts = run_dir / "trainer_state.json"
    if ts.exists():
        return json.loads(ts.read_text())
    return None


def _extract_run(run_dir: Path) -> dict:
    ts = _load_trainer_state(run_dir)
    name = run_dir.name
    if ts is None:
        return {
            "run": name,
            "status": "no_trainer_state",
        }
    log = ts.get("log_history") or []
    train_entries = [e for e in log if "loss" in e and "eval_loss" not in e]
    eval_entries = [e for e in log if "eval_loss" in e]

    last_train_loss = train_entries[-1]["loss"] if train_entries else None
    min_train_loss = (min(e["loss"] for e in train_entries)
                      if train_entries else None)
    max_grad = (max((e.get("grad_norm") or 0.0) for e in train_entries)
                if train_entries else None)
    last_grad = train_entries[-1].get("grad_norm") if train_entries else None

    best_eval = None
    best_eval_step = None
    last_eval = None
    last_eval_step = None
    last_eval_acc = None
    if eval_entries:
        best = min(eval_entries, key=lambda e: e.get("eval_loss", math.inf))
        best_eval = best.get("eval_loss")
        best_eval_step = best.get("step")
        last = eval_entries[-1]
        last_eval = last.get("eval_loss")
        last_eval_step = last.get("step")
        last_eval_acc = last.get("eval_assistant_token_acc")

    # lr from sft_config (fallback: parse tag "lr3e-4_r64")
    lr = None
    lora_r = None
    cfg_path = run_dir / "sft_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        lr = cfg.get("lr")
        lora_r = cfg.get("lora_r")
    if lr is None and name.startswith("lr"):
        tag = name.split("_")[0][2:]
        try:
            lr = float(tag)
        except ValueError:
            lr = None
    if lora_r is None and "_r" in name:
        try:
            lora_r = int(name.split("_r")[-1])
        except ValueError:
            lora_r = None

    return {
        "run": name,
        "lr": lr,
        "lora_r": lora_r,
        "epoch": ts.get("epoch"),
        "global_step": ts.get("global_step"),
        "max_steps": ts.get("max_steps"),
        "last_train_loss": last_train_loss,
        "min_train_loss": min_train_loss,
        "max_grad_norm": max_grad,
        "last_grad_norm": last_grad,
        "best_eval_loss": best_eval,
        "best_eval_step": best_eval_step,
        "last_eval_loss": last_eval,
        "last_eval_step": last_eval_step,
        "last_eval_token_acc": last_eval_acc,
        "n_eval_events": len(eval_entries),
        "needs_backfill": len(eval_entries) == 0,
    }


def _fmt(v, fmt: str = "{:.4f}") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return fmt.format(v)
    return str(v)


def _render_markdown(rows: list[dict], sweep_dir: Path) -> str:
    any_eval = any(r.get("best_eval_loss") is not None for r in rows)
    ranked = sorted(
        rows,
        key=lambda r: (
            r.get("best_eval_loss") is None,
            r.get("best_eval_loss") if r.get("best_eval_loss") is not None
            else r.get("min_train_loss") or math.inf,
        ),
    )
    header = (
        "| rank | run | lr | r | epoch | steps | best_eval | best_step | "
        "last_eval | token_acc | min_train | last_train | max |∇| |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for i, r in enumerate(ranked, 1):
        body.append(
            "| {rank} | {run} | {lr} | {r} | {ep} | {step} | {best_eval} | "
            "{best_step} | {last_eval} | {acc} | {min_train} | {last_train} | "
            "{grad} |".format(
                rank=i,
                run=r.get("run", "?"),
                lr=_fmt(r.get("lr"), "{:.0e}"),
                r=_fmt(r.get("lora_r"), "{}"),
                ep=_fmt(r.get("epoch"), "{:.2f}"),
                step=_fmt(r.get("global_step"), "{}"),
                best_eval=_fmt(r.get("best_eval_loss")),
                best_step=_fmt(r.get("best_eval_step"), "{}"),
                last_eval=_fmt(r.get("last_eval_loss")),
                acc=_fmt(r.get("last_eval_token_acc")),
                min_train=_fmt(r.get("min_train_loss")),
                last_train=_fmt(r.get("last_train_loss")),
                grad=_fmt(r.get("max_grad_norm"), "{:.2f}"),
            )
        )
    warn = ""
    if not any_eval:
        warn = (
            "\n> ⚠️ **No eval events were recorded in any run.** This is the "
            "classic `eval_steps > max_steps` bug. Backfill with:\n>\n"
            "> ```bash\n"
            "> python -m src.training.evaluate_sweep \\\n"
            f">     --sweep_dir {sweep_dir} \\\n"
            ">     --val_file  outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl\n"
            "> ```\n"
        )
    return (
        f"# Phase C SFT sweep summary\n\n"
        f"- sweep_dir: `{sweep_dir}`\n"
        f"- runs: {len(rows)}\n"
        f"- runs needing eval backfill: "
        f"{sum(1 for r in rows if r.get('needs_backfill'))}\n"
        f"{warn}\n"
        f"## Ranked by best_eval_loss (fallback: min_train_loss)\n\n"
        f"{header}{chr(10).join(body)}\n"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_dir", required=True, help="Parent dir holding sweep runs.")
    p.add_argument("--out", default=None, help="Output markdown path.")
    p.add_argument("--out_json", default=None, help="Output JSON path.")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs = [d for d in sorted(sweep_dir.iterdir()) if d.is_dir()]
    rows = [_extract_run(d) for d in runs if (d / "sft_config.json").exists()
            or any(d.glob("checkpoint-*"))]

    md = _render_markdown(rows, sweep_dir)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"[summarize_sweep] wrote {args.out}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"[summarize_sweep] wrote {args.out_json}")


if __name__ == "__main__":
    main()
