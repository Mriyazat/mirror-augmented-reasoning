"""Build direction-only AB/BA preference pairs from a mirror SFT corpus.

This is deliberately narrower than `build_preference_pairs.py`.

The failed mirror-SFT experiment showed that mixing mirror augmentation into
normal SFT can damage free-running JSON/classification. The specific remaining
failure in the good v1 student is directional mirror behavior, so this builder
creates only the preference signal needed for that:

    prompt   = AB or BA prompt from the mirror corpus
    chosen   = teacher assistant JSON with the correct direction_tag
    rejected = same assistant JSON with a_to_b <-> b_to_a flipped

Bidirectional and abstention records are skipped. The result is suitable for a
small IPO/DPO stage starting from the known-good SFT adapter.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


_DIR_FLIP = {"a_to_b": "b_to_a", "b_to_a": "a_to_b"}


def _assistant_content(rec: dict) -> str | None:
    for msg in rec.get("messages") or []:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return None


def _prompt_messages(rec: dict) -> list[dict]:
    return [m for m in rec.get("messages") or [] if m.get("role") != "assistant"]


def _flip_trace_direction(trace: dict) -> dict | None:
    """Return a rejected trace with directional tags flipped.

    We only build pairs for final_answer.direction_tag in {a_to_b, b_to_a}.
    Step-level matching directional tags are flipped too. Other fields are
    intentionally unchanged so the preference is about direction only.
    """
    fa = trace.get("final_answer")
    if not isinstance(fa, dict):
        return None
    tag = fa.get("direction_tag")
    flipped = _DIR_FLIP.get(tag)
    if flipped is None:
        return None

    rejected = copy.deepcopy(trace)
    rejected["final_answer"]["direction_tag"] = flipped
    for step in rejected.get("steps") or []:
        if not isinstance(step, dict):
            continue
        st = step.get("direction_tag")
        if st in _DIR_FLIP:
            step["direction_tag"] = _DIR_FLIP[st]
    return rejected


def build(input_path: Path, output_path: Path,
          include_tiers: set[str]) -> dict:
    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stats["input"] += 1

            tier = rec.get("tier")
            if tier not in include_tiers:
                stats[f"skip_tier:{tier}"] += 1
                continue

            content = _assistant_content(rec)
            if not content:
                stats["skip_missing_assistant"] += 1
                continue
            try:
                trace = json.loads(content)
            except Exception:
                stats["skip_bad_assistant_json"] += 1
                continue

            if (trace.get("final_answer") or {}).get("abstain"):
                stats["skip_abstain"] += 1
                continue

            rejected = _flip_trace_direction(trace)
            if rejected is None:
                stats["skip_non_directional"] += 1
                continue

            pref = {
                "pair_id": rec.get("pair_id") or "",
                "mirror_type": "direction_flip_abba",
                "input_order": rec.get("input_order", "unknown"),
                "prompt": _prompt_messages(rec),
                "chosen": json.dumps(trace, ensure_ascii=False, separators=(",", ":")),
                "rejected": json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
                "prm_chosen": None,
                "prm_rejected": None,
                "source_tier": tier,
            }
            fout.write(json.dumps(pref, ensure_ascii=False) + "\n")
            stats["output"] += 1
            stats[f"output_order:{pref['input_order']}"] += 1
            stats[f"output_tier:{tier}"] += 1
            stats[f"output_dir:{trace['final_answer'].get('direction_tag')}"] += 1

    return dict(stats)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="Mirror SFT corpus JSONL, e.g. train.mirror.jsonl")
    p.add_argument("--output", required=True,
                   help="Direction-only preference JSONL to write")
    p.add_argument("--include_tiers", default="full_correct,family_correct",
                   help="Comma-separated SFT tiers to include")
    args = p.parse_args()

    include_tiers = {x.strip() for x in args.include_tiers.split(",") if x.strip()}
    stats = build(Path(args.input), Path(args.output), include_tiers)
    print(f"[direction_prefs] wrote {args.output}")
    for k, v in sorted(stats.items()):
        print(f"  {k:28} {v}")


if __name__ == "__main__":
    main()
