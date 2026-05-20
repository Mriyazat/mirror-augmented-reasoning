"""Trace-coherence rescuer.

For each student prediction whose `trace.steps[*].family_hint` strongly
disagrees with `final_answer.family`, OVERRIDE the final family with the
trace-majority hint. Two policies:

  hint_majority    — strict plurality vote over non-conclusion family_hints.
                     Requires at least `min_steps` votes and that the
                     plurality beats the final_answer's family.
  conclusion_text  — regex over the conclusion step's `claim` text for
                     each of the 7 families; trust the text over the field.
  hybrid           — apply hint_majority first; if no override is triggered,
                     apply conclusion_text. Records the rescue rule used.

A predicted record is overridden ONLY if BOTH:
  (a) the trace's signal is strong (>= `min_strength` fraction of votes), and
  (b) the original `final_answer.family` is missing from the trace hints
      OR is a strict minority compared to the rescued family.

Writes a new JSONL with `final_prediction.family` replaced, preserving
all other fields. Adds `rescue_rule` and `rescue_from` markers for audit.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]

# Hand-written regex hints over conclusion text. Order matters; more specific
# patterns first.
TEXT_PATTERNS = [
    ("PK_Absorption",   re.compile(r"\b(absorption|chelat\w*|bind\w* in (?:the )?gut|reduce[sd]? absorption|impair\w* absorption)\b", re.I)),
    ("PK_Excretion",    re.compile(r"\b(excretion|renal cleara\w+|tubular secretion|elimination|reabsorption|cleara\w+ via the kidney)\b", re.I)),
    ("PK_Distribution", re.compile(r"\b(serum concentration|plasma concentration|protein binding|displac\w+ from albumin|distribution)\b", re.I)),
    ("PK_Metabolism",   re.compile(r"\b(CYP\d|CYP[1-9][A-Z]\d?|induc\w+|inhibit\w+ (?:CYP|metabolism)|metaboliz\w+|metabolism|first[- ]pass)\b", re.I)),
    ("AdverseRisk",     re.compile(r"\b(risk|adverse|toxic\w+|QTc|prolonga\w+|bleeding|hypoglyc\w+|hyperten\w+|hypoten\w+|CNS depress\w+|sedat\w+|neuromuscular|nephrotox\w+|hepatotox\w+|cardio[- ]?toxic\w*)\b", re.I)),
    ("Efficacy",        re.compile(r"\b(therapeutic efficacy|effectiveness|reduce[sd]? (?:the )?effect|enhance[sd]? (?:the )?effect|antagoniz\w+ effect|potentiat\w+)\b", re.I)),
    ("PD_Activity",     re.compile(r"\b(antihyperten\w+|hypotens\w+|sympatholy\w+|sympathomimet\w+|vasocons\w+|vasodilat\w+|sedat\w+ effect|arrhythm\w+|antiarrhythm\w+|anticoag\w+|antiplate\w+)\b", re.I)),
]


def hint_majority(steps: list[dict], original: str, min_steps: int, min_strength: float,
                  max_original_frac: float = 1.0):
    hints = [s.get("family_hint") for s in steps if s.get("role") != "conclusion"]
    hints = [h for h in hints if h in FAMS]
    if len(hints) < min_steps:
        return None
    c = Counter(hints)
    winner, n_winner = c.most_common(1)[0]
    if winner == original:
        return None
    strength = n_winner / len(hints)
    if strength < min_strength:
        return None
    n_original = c.get(original, 0)
    if n_original >= n_winner:
        return None
    if (n_original / len(hints)) > max_original_frac:
        return None
    return winner


def conclusion_text(steps: list[dict], original: str):
    if not steps:
        return None
    last = steps[-1]
    if last.get("role") != "conclusion":
        return None
    text = last.get("claim") or ""
    for fam, pat in TEXT_PATTERNS:
        if pat.search(text):
            if fam != original:
                return fam
            else:
                return None
    return None


def rescue(record: dict, policy: str, min_steps: int, min_strength: float,
           max_conf: float = 1.0, max_original_frac: float = 1.0):
    trace = record.get("trace") or {}
    steps = trace.get("steps") or []
    fa = record.get("final_prediction") or {}
    original = fa.get("family")
    if not original:
        return None, None
    # Confidence gate: only rescue uncertain predictions
    try:
        conf = float(fa.get("confidence", 1.0) or 1.0)
    except Exception:
        conf = 1.0
    if conf > max_conf:
        return None, None

    if policy == "hint_majority":
        new = hint_majority(steps, original, min_steps, min_strength, max_original_frac)
        return ("hint_majority", new) if new else (None, None)
    if policy == "conclusion_text":
        new = conclusion_text(steps, original)
        return ("conclusion_text", new) if new else (None, None)
    if policy == "hybrid":
        new = hint_majority(steps, original, min_steps, min_strength, max_original_frac)
        if new:
            return ("hint_majority", new)
        new = conclusion_text(steps, original)
        if new:
            return ("conclusion_text", new)
        return (None, None)
    raise ValueError(f"unknown policy {policy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--policy", choices=["hint_majority", "conclusion_text", "hybrid"], default="hybrid")
    ap.add_argument("--min_steps", type=int, default=2,
                    help="Minimum non-conclusion family_hints to consider a rescue.")
    ap.add_argument("--min_strength", type=float, default=0.5,
                    help="Plurality strength required to override (fraction of votes).")
    ap.add_argument("--max_conf", type=float, default=1.0,
                    help="Only rescue predictions whose confidence is <= this (default: all).")
    ap.add_argument("--max_original_frac", type=float, default=1.0,
                    help="Block rescue if original family share of hints > this (default: none).")
    args = ap.parse_args()

    n_total = n_rescued = 0
    by_rule = Counter()
    by_trans = Counter()
    op = Path(args.output)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, op.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            rule, new = rescue(
                r, args.policy, args.min_steps, args.min_strength,
                max_conf=args.max_conf, max_original_frac=args.max_original_frac,
            )
            if new:
                old = (r.get("final_prediction") or {}).get("family")
                r.setdefault("rescue", {})
                r["rescue"]["rule"] = rule
                r["rescue"]["from"] = old
                r["rescue"]["to"] = new
                r["final_prediction"]["family"] = new
                n_rescued += 1
                by_rule[rule] += 1
                by_trans[(old, new)] += 1
            fout.write(json.dumps(r) + "\n")

    print(f"[trace_rescue] {n_rescued}/{n_total} predictions rescued ({100*n_rescued/max(1,n_total):.1f}%)")
    print(f"[trace_rescue] by rule: {dict(by_rule)}")
    print("[trace_rescue] top transitions:")
    for (o, n), k in by_trans.most_common(10):
        print(f"  {o:>16s} -> {n:<16s}  {k}")
    print(f"[trace_rescue] wrote {op}")


if __name__ == "__main__":
    main()
