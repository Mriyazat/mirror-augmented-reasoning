"""Build student inference prompt JSONL files for held-out split evaluation.

`src.inference.predict` consumes the Phase-C SFT JSONL shape:

    {"pair_id": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}]}

The Phase 4 validation file already exists in that format, but the cold-split
test manifests are just pair IDs. This utility rebuilds the deterministic
teacher prompt for each manifest pair without calling any teacher model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.prompt import build_prompt


def _load_manifest(path: Path, split_section: str) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    return [r for r in rows if r.get("split") == split_section]


def build(split: str, split_section: str, output: Path, limit: int = 0,
          neighbor_index_path: Path | None = None,
          manifest_jsonl: Path | None = None) -> dict:
    """Build student eval prompts.

    Args:
      split: tag for default split manifest lookup
      split_section: train/val/test
      output: where to write the prompt JSONL
      limit: optional cap on number of pairs (use a stratified manifest
             instead; this just truncates from the top)
      neighbor_index_path: explicit path to a precomputed neighbor index
             parquet. If None, falls back to data_processed/neighbor_index_<split>.parquet.
      manifest_jsonl: optional JSONL with `pair_id` rows. If provided,
             overrides split/split_section manifest selection (useful for a
             stratified pre-built sample).
    """
    if manifest_jsonl is not None:
        rows = []
        with manifest_jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rows.append({"pair_id": rec["pair_id"], "split": split_section})
        print(f"[eval_prompts] using custom manifest: {manifest_jsonl} ({len(rows):,} pairs)")
    else:
        manifest_path = DATA / "splits" / f"manifest_{split}.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing split manifest: {manifest_path}")
        rows = _load_manifest(manifest_path, split_section)

    if limit > 0:
        rows = rows[:limit]

    if neighbor_index_path is None:
        neighbor_index_path = DATA / f"neighbor_index_{split}.parquet"
    if neighbor_index_path.exists():
        print(f"[eval_prompts] using neighbor index: {neighbor_index_path}")
    else:
        print(f"[eval_prompts] WARNING: no neighbor index at "
              f"{neighbor_index_path}; prompts will have empty neighbor block "
              "(distribution mismatch with training!)")
    cb = ContextBuilder(neighbor_index_path=neighbor_index_path
                        if neighbor_index_path.exists() else None)

    output.parent.mkdir(parents=True, exist_ok=True)
    stats = {"input": len(rows), "written": 0, "errors": 0}
    err_path = output.with_suffix(output.suffix + ".errors.jsonl")
    with output.open("w") as fout, err_path.open("w") as ferr:
        for i, row in enumerate(rows, start=1):
            pid = row["pair_id"]
            try:
                ctx = cb.build(pid)
                prompt = build_prompt(ctx)
                rec = {
                    "pair_id": pid,
                    "split": split,
                    "split_section": split_section,
                    "context_ids": sorted(ctx.context_ids()),
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    "prompt_version": prompt["prompt_version"],
                    "prompt_sha": prompt["prompt_sha"],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["written"] += 1
            except Exception as e:
                ferr.write(json.dumps({
                    "pair_id": pid,
                    "error": f"{type(e).__name__}: {e}",
                }) + "\n")
                stats["errors"] += 1
            if i % 100 == 0 or i == len(rows):
                print(
                    f"[eval_prompts] {split}/{split_section}: "
                    f"{i:,}/{len(rows):,} written={stats['written']:,} "
                    f"errors={stats['errors']:,}",
                    flush=True,
                )
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True,
                   choices=["random_full", "drug_cold", "pair_cold", "subset25k"])
    p.add_argument("--split_section", default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--neighbor_index_path", type=str, default=None,
                   help="Explicit path to a precomputed neighbor index "
                        "parquet (overrides default data_processed/"
                        "neighbor_index_<split>.parquet).")
    p.add_argument("--manifest_jsonl", type=str, default=None,
                   help="Optional JSONL of {pair_id} rows to use as the query "
                        "manifest (overrides split-based selection).")
    args = p.parse_args()

    stats = build(
        split=args.split,
        split_section=args.split_section,
        output=Path(args.output),
        limit=args.limit,
        neighbor_index_path=Path(args.neighbor_index_path) if args.neighbor_index_path else None,
        manifest_jsonl=Path(args.manifest_jsonl) if args.manifest_jsonl else None,
    )
    print(f"[eval_prompts] wrote {args.output}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
