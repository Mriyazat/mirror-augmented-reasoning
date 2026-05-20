"""Build a mirror-augmented SFT training corpus.

For each canonical teacher record, emit TWO records:
  - input_order="ab" : the record unchanged (canonical AB)
  - input_order="ba" : the record's user prompt structurally swapped
                       A<->B (via `src.inference.predict._mirror_user_content`)
                       and the assistant trace's `direction_tag` flipped
                       (a_to_b <-> b_to_a; bidirectional / n/a unchanged).

Both records share:
  - pair_id              (canonical DB_a|DB_b)
  - mirror_pair_id = pair_id (tells the SFT trainer these are a mirror set)
  - family, subtype, polarity, tier, sample_weight   (identity-preserving)

Why
---
`src/training/sft_train.py`'s symmetry-KL term fires only when a batch
contains both AB and BA records of the same canonical pair. A teacher
corpus with one record per pair_id leaves the symmetry coefficient
inactive at the loss level; this script materializes both orderings so
the symmetry-KL term contributes during training.

Invariants
----------
1. Mirror is involutive: mirror(mirror(x)) == x on the user prompt.
2. final_answer.direction_tag flips: a_to_b <-> b_to_a.
3. step[i].direction_tag flips: a_to_b <-> b_to_a.
4. final_answer.family / subtype / polarity and each step's family_hint /
   claim text / evidence_ids are role-agnostic and stay unchanged:
     - Drug names in claims (e.g. "Benzthiazide inhibits ...") are
       chemistry-identity, not role-identity; they remain correct
       regardless of which prompt slot the drug is presented in.
     - Polarity describes effect magnitude (up/down) and does not flip
       under role swap.
5. Abstention records: the user prompt is still mirrored (so the student
   sees the same ambiguous prompt in both orderings), and the assistant
   trace is unchanged (an abstention stays an abstention). This teaches
   AB/BA abstention consistency.

Usage
-----
    python -m src.data.build_mirror_sft_corpus \\
        --input  outputs/phase_c/teacher_clean.reasoning_safe.train.jsonl \\
        --output outputs/phase_c/teacher_clean.reasoning_safe.train.mirror.jsonl \\
        --scope all                 # all | committed_only

Output JSONL one record per line. Each record has all the fields of
the input PLUS `input_order`, `mirror_pair_id`. Downstream code in
`src.training.sft_train.SftJsonlDataset` already expects those fields.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Reuse the tested mirror logic from the inference module. It lives in
# predict.py because predict.py itself needs to synthesize BA prompts from
# canonical AB records; this script does the same rewrite on the training
# corpus user turn. Keeping a single source of truth prevents drift between
# predict-time and train-time mirror semantics.
from src.inference.predict import _mirror_user_content


_DIR_FLIP = {"a_to_b": "b_to_a", "b_to_a": "a_to_b"}


def _flip_direction(tag):
    """Flip a_to_b <-> b_to_a; leave bidirectional / n/a / None alone."""
    return _DIR_FLIP.get(tag, tag)


def _mirror_assistant_json(content: str) -> tuple[str, bool]:
    """Return a mirrored version of the assistant's JSON trace.

    Flips direction_tag in final_answer and in every step (only a_to_b
    <-> b_to_a; bidirectional / n/a are untouched). Claim text and
    evidence_ids stay put -- they are role-agnostic.

    Returns (new_content, ok). If the content is not valid JSON we return
    it unchanged with ok=False so the caller can decide to drop the record
    or emit an ab-only copy.
    """
    try:
        obj = json.loads(content)
    except Exception:
        return content, False

    # final_answer
    fa = obj.get("final_answer")
    if isinstance(fa, dict):
        fa["direction_tag"] = _flip_direction(fa.get("direction_tag"))

    # per-step direction_tag
    steps = obj.get("steps") or []
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and "direction_tag" in s:
                s["direction_tag"] = _flip_direction(s["direction_tag"])

    # Separator choice must match the teacher corpus style so the token
    # sequence stays close to the original. prepare_phase_c writes with
    # `separators=(",", ":")`; we match that.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")), True


def _mirror_record(rec: dict,
                   stats: Counter) -> dict | None:
    """Produce the BA-order mirror of a single canonical record.

    Returns None when mirroring fails (non-parseable user prompt or
    assistant JSON). The caller should still emit the AB-order record
    either way; we never lose training data because of a mirror failure.
    """
    pid = rec.get("pair_id") or ""
    if "|" not in pid:
        stats["skipped_no_pipe"] += 1
        return None

    msgs = rec.get("messages") or []
    new_msgs = []
    user_mirrored = False
    assistant_mirrored = False
    abstain_record = (rec.get("tier") == "abstention")

    for m in msgs:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            new_content, ok, _ = _mirror_user_content(content, pid)
            if not ok:
                stats["skipped_user_mirror_fail"] += 1
                return None
            new_msgs.append({"role": "user", "content": new_content})
            user_mirrored = True
        elif role == "assistant":
            if abstain_record:
                # An abstain response is already role-invariant (we emit
                # the same "I can't answer" JSON regardless of ordering).
                new_msgs.append(m)
                assistant_mirrored = True
                continue
            new_content, ok = _mirror_assistant_json(content)
            if not ok:
                stats["skipped_assistant_mirror_fail"] += 1
                return None
            new_msgs.append({"role": "assistant", "content": new_content})
            assistant_mirrored = True
        else:
            new_msgs.append(m)

    if not user_mirrored or not assistant_mirrored:
        stats["skipped_missing_role"] += 1
        return None

    out = dict(rec)
    out["messages"] = new_msgs
    out["input_order"] = "ba"
    out["mirror_pair_id"] = pid

    # Flip direction_tag at the top level for ALL records (including
    # abstentions). This field describes the gold direction relative to
    # the ordering in which the pair is PRESENTED; it is not the
    # student's target. The student's target is the assistant JSON
    # (unchanged for abstentions; direction-flipped for commitments).
    # Keeping the metadata coherent matters for any downstream consumer
    # that groups or filters by direction.
    top_dir = rec.get("direction_tag")
    out["direction_tag"] = _flip_direction(top_dir) if top_dir else top_dir

    return out


def _self_test_involution(records_sample: list[dict], n_to_test: int = 50) -> int:
    """Check mirror(mirror(x)) == x for user prompts on a sample.

    Returns the number of records that failed the involution property.
    Prints a short diagnostic. This is a data-integrity check, not a
    correctness proof of the surrounding JSON fields.
    """
    n_fail = 0
    for rec in records_sample[:n_to_test]:
        pid = rec.get("pair_id") or ""
        if "|" not in pid:
            continue
        user_msg = next((m for m in rec.get("messages", []) if m.get("role") == "user"), None)
        if user_msg is None:
            continue
        orig = user_msg.get("content") or ""
        once, ok1, _ = _mirror_user_content(orig, pid)
        if not ok1:
            continue
        twice, ok2, _ = _mirror_user_content(once, pid)
        if not ok2 or twice != orig:
            n_fail += 1
    return n_fail


def build(input_path: Path, output_path: Path, scope: str = "all",
          self_test_n: int = 50) -> dict:
    stats: Counter = Counter()
    n_in = 0
    n_ab = 0
    n_ba = 0
    records_sample: list[dict] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_in += 1
            if len(records_sample) < self_test_n:
                records_sample.append(rec)

            # Always emit AB (original) record, annotated
            ab = dict(rec)
            ab["input_order"] = "ab"
            ab["mirror_pair_id"] = rec.get("pair_id")
            fout.write(json.dumps(ab, ensure_ascii=False,
                                  separators=(",", ":")) + "\n")
            n_ab += 1
            stats[f"ab_{rec.get('tier','?')}"] += 1

            # Scope check for the BA emission
            if scope == "committed_only" and rec.get("tier") == "abstention":
                stats["scope_skip_abstention"] += 1
                continue

            ba = _mirror_record(rec, stats)
            if ba is None:
                continue
            fout.write(json.dumps(ba, ensure_ascii=False,
                                  separators=(",", ":")) + "\n")
            n_ba += 1
            stats[f"ba_{rec.get('tier','?')}"] += 1

    # Involution sanity check
    n_inv_fail = _self_test_involution(records_sample, self_test_n)

    return {
        "input_records":   n_in,
        "output_records":  n_ab + n_ba,
        "ab_records":      n_ab,
        "ba_records":      n_ba,
        "scope":           scope,
        "n_involution_fail": n_inv_fail,
        "n_involution_tested": min(self_test_n, n_in),
        "detail":          dict(stats),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="teacher_clean.reasoning_safe.train.jsonl")
    p.add_argument("--output", required=True,
                   help="mirror-augmented train corpus JSONL")
    p.add_argument("--scope", default="all",
                   choices=["all", "committed_only"],
                   help="all = mirror every record incl abstentions; "
                        "committed_only = skip BA for abstention-tier records")
    p.add_argument("--self_test_n", type=int, default=50,
                   help="sample size for the mirror-involution check")
    args = p.parse_args()

    out = build(Path(args.input), Path(args.output),
                scope=args.scope, self_test_n=args.self_test_n)

    print("[mirror_corpus] wrote", args.output)
    print(f"  input records          : {out['input_records']:,}")
    print(f"  output records         : {out['output_records']:,}")
    print(f"  AB records             : {out['ab_records']:,}")
    print(f"  BA records             : {out['ba_records']:,}")
    print(f"  scope                  : {out['scope']}")
    print(f"  involution_fail        : {out['n_involution_fail']}/{out['n_involution_tested']}")
    for k, v in sorted(out["detail"].items()):
        print(f"  {k:32} {v}")

    if out["n_involution_fail"] > 0:
        print("\n[mirror_corpus] WARNING: some user prompts failed the "
              "mirror(mirror(x)) == x check. Inspect output before training.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
