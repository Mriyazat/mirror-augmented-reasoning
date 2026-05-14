"""B7.2 -- Programmatic DrugBank-description-vs-trace cross-check.

For every trace in teacher_clean.jsonl that COMMITS (non-abstention),
compare its declared answer (family, subtype, polarity, direction_tag,
summary) against the gold-standard text in labels_hierarchical.parquet.

Four axes
---------
  A.  family_match            (exact, gold vs trace)
  B.  subtype_keyword_match   (gold subtype keyword appears in summary)
  C.  polarity_match          (compatible: up/down/risk/risk_down)
  D.  direction_match         (canonical direction inferred from subject id)

A trace passes the cross-check when A & C & D are all true.
B is informational (subtype is a finer label that's hard to require).

Per-tier numbers:
  full_correct -- expect >=95% on A & C & D
  family_correct -- expect 100% on A, then partial on C/D
  near_miss -- expect 100% on A by definition, lower on C/D
  abstention -- excluded (no commit)

Outputs
-------
  drugbank_crosscheck_summary.md   -- aggregate report, tier x axis
  drugbank_crosscheck_failures.jsonl -- one record per FAIL trace, with
                                       gold/trace fields side-by-side

CLI
---
    python -m src.teacher.drugbank_crosscheck \
        --teacher_clean /path/to/teacher_clean.jsonl \
        --labels        data_processed/labels_hierarchical.parquet \
        --output_dir    outputs/audit/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.teacher.schema import extract_json_block


# ---------- Canonical gold-direction inference ----------
def _canonical_direction(rec: dict) -> str | None:
    """Return one of {a_to_b, b_to_a, bidirectional, None}.

    Inferred from labels_hierarchical row: subject_drugbank_id +
    pair_id (a_id|b_id) + bidirectional flag.
    """
    if rec.get("bidirectional"):
        return "bidirectional"
    a_id, b_id = rec["pair_id"].split("|")
    sub = rec.get("subject_drugbank_id")
    if not sub:
        return None
    if sub == a_id:
        return "a_to_b"
    if sub == b_id:
        return "b_to_a"
    return None


def _polarity_match(gold_pol: str | None, trace_pol: str | None) -> bool:
    """Check polarity compatibility.

    Map gold values:
      up        -> {up}
      down      -> {down}
      risk      -> {risk, up}     (some traces tag risk_up as 'up')
      risk_down -> {down, risk_down}
      None      -> any (cannot test)
    """
    if gold_pol is None:
        return True
    g = gold_pol.lower()
    t = (trace_pol or "").lower()
    if g == "up":
        return t in ("up", "risk")
    if g == "down":
        return t in ("down",)
    if g == "risk":
        return t in ("risk", "up")
    if g == "risk_down":
        # "risk_down" means severity/risk is decreased.  A bare "risk"
        # trace usually means increased risk, so do not accept it here.
        return t in ("down", "risk_down")
    return g == t


# ---------- Subtype-keyword extraction ----------
_SUBTYPE_STOPWORDS = {"and", "or", "of", "the", "a", "an"}


def _subtype_keywords(subtype: str | None) -> list[str]:
    if not subtype:
        return []
    parts = re.split(r"[_\s\-]+", subtype.lower())
    return [p for p in parts if p and p not in _SUBTYPE_STOPWORDS and len(p) > 2]


def _summary_text(parsed: dict) -> str:
    fa = parsed.get("final_answer") or {}
    summ = (fa.get("summary") or "").lower()
    # Concatenate all step claims for richer matching
    claims = " ".join(
        (s.get("claim") or "").lower() for s in (parsed.get("steps") or [])
    )
    return f"{summ} || {claims}"


def _subtype_keyword_match(subtype: str | None, summary: str) -> tuple[bool, list[str]]:
    kws = _subtype_keywords(subtype)
    if not kws:
        return True, []
    hits = [k for k in kws if k in summary]
    return (len(hits) >= 1), hits


# ---------- Description-keyword check (descriptive, optional axis) ----------
_DESCRIPTION_KEYWORDS = {
    "metabolism", "absorption", "excretion", "serum",
    "concentration", "anticoagulant", "antihypertensive", "hypotensive",
    "hypertensive", "sedative", "analgesic", "cns", "depression",
    "depressant", "bleeding", "hemorrhage", "qtc", "tachycardia",
    "bradycardia", "neuroexcitatory", "vasodilatory", "vasoconstricting",
    "neuromuscular", "cardiotoxic", "hepatotoxic", "nephrotoxic",
    "constipation", "diarrhea", "hypoglycemia", "hyperkalemia",
    "hypokalemia", "thrombosis", "myopathy", "rhabdomyolysis",
    "serotonergic", "anticholinergic", "antiplatelet", "diuretic",
    "immunosuppressive", "bioavailability", "protein", "binding",
    "metabolite",
}


def _description_keyword_overlap(description: str | None, summary: str) -> tuple[float, list[str]]:
    if not description:
        return 1.0, []
    desc_words = set(re.findall(r"[a-z]+", description.lower()))
    relevant_in_desc = desc_words & _DESCRIPTION_KEYWORDS
    if not relevant_in_desc:
        return 1.0, []
    hits = [w for w in relevant_in_desc if w in summary]
    return (len(hits) / len(relevant_in_desc)), hits


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--labels",        required=True)
    ap.add_argument("--output_dir",    required=True)
    args = ap.parse_args()

    # 1) Load gold
    print(f"[crosscheck] loading labels...")
    tbl = pq.read_table(args.labels, columns=[
        "pair_id", "description", "template", "family", "subtype",
        "polarity", "subject_drugbank_id", "object_drugbank_id",
        "bidirectional",
    ]).to_pylist()
    gold: dict[str, dict] = {r["pair_id"]: r for r in tbl}
    print(f"[crosscheck] {len(gold):,} gold rows loaded")

    # 2) Load traces
    print(f"[crosscheck] loading teacher_clean...")
    traces: list[dict] = []
    for line in open(args.teacher_clean):
        traces.append(json.loads(line))
    print(f"[crosscheck] {len(traces):,} traces loaded")

    # 3) For each trace, compute axes
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fail_path = out_dir / "drugbank_crosscheck_failures.jsonl"
    fail_fh = open(fail_path, "w")

    # Aggregate counters: tier -> axis -> {pass, fail, n_applicable}
    agg: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"pass": 0, "fail": 0, "n": 0})
    )
    fam_agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "all_pass": 0, "any_fail": 0}
    )

    n_skip_no_gold = 0
    n_abstain = 0
    n_no_parse = 0
    n_committed = 0

    for rec in traces:
        pid = rec["pair_id"]
        tier = rec.get("tier", "?")
        if pid not in gold:
            n_skip_no_gold += 1
            continue
        g = gold[pid]

        msg = rec["messages"][2]["content"] if len(rec["messages"]) > 2 else ""
        parsed = extract_json_block(msg)
        if parsed is None:
            n_no_parse += 1
            continue

        fa = parsed.get("final_answer") or {}
        if fa.get("abstain", False) or tier == "abstention":
            n_abstain += 1
            continue
        n_committed += 1

        family_g  = (g.get("family") or "").strip()
        subtype_g = (g.get("subtype") or "").strip()
        pol_g     = (g.get("polarity") or "").strip() or None
        dir_g     = _canonical_direction(g)

        family_t  = (fa.get("family") or "").strip()
        subtype_t = (fa.get("subtype") or "").strip()
        pol_t     = (fa.get("polarity") or "").strip() or None
        dir_t     = (fa.get("direction_tag") or "").strip() or None

        summary = _summary_text(parsed)

        # axis A: family
        a_pass = (family_g == family_t and family_g != "")
        # axis B: subtype keyword
        b_pass, b_hits = _subtype_keyword_match(subtype_g, summary)
        # axis C: polarity
        c_pass = _polarity_match(pol_g, pol_t)
        # axis D: direction
        # NOTE: when gold is bidirectional, any committed direction
        # (a_to_b / b_to_a / bidirectional) is acceptable -- the trace
        # is being MORE specific than gold, not contradicting it.
        if dir_g is None:
            d_pass = True
            d_applicable = False
        elif dir_g == "bidirectional":
            d_pass = (dir_t in ("a_to_b", "b_to_a", "bidirectional")) if dir_t else False
            d_applicable = True
        else:
            d_pass = (dir_t == dir_g) if dir_t else False
            d_applicable = True
        # axis E: description keyword overlap (informational)
        kw_overlap, kw_hits = _description_keyword_overlap(g.get("description"), summary)

        agg[tier]["family"]["pass"]    += int(a_pass);  agg[tier]["family"]["fail"]    += int(not a_pass);  agg[tier]["family"]["n"]    += 1
        agg[tier]["subtype_kw"]["pass"]+= int(b_pass);  agg[tier]["subtype_kw"]["fail"]+= int(not b_pass);  agg[tier]["subtype_kw"]["n"]+= 1
        agg[tier]["polarity"]["pass"]  += int(c_pass);  agg[tier]["polarity"]["fail"]  += int(not c_pass);  agg[tier]["polarity"]["n"]  += 1
        if d_applicable:
            agg[tier]["direction"]["pass"] += int(d_pass); agg[tier]["direction"]["fail"] += int(not d_pass); agg[tier]["direction"]["n"] += 1

        all_pass = a_pass and c_pass and (d_pass if d_applicable else True)
        fam_agg[family_g]["n"] += 1
        if all_pass:
            fam_agg[family_g]["all_pass"] += 1
        else:
            fam_agg[family_g]["any_fail"] += 1
            fail_fh.write(json.dumps({
                "pair_id":         pid,
                "tier":            tier,
                "axes": {
                    "family":     {"pass": a_pass, "gold": family_g,  "trace": family_t},
                    "subtype_kw": {"pass": b_pass, "gold_subtype": subtype_g, "kw_hits": b_hits},
                    "polarity":   {"pass": c_pass, "gold": pol_g,     "trace": pol_t},
                    "direction":  {"pass": d_pass if d_applicable else None,
                                   "gold": dir_g, "trace": dir_t},
                },
                "kw_overlap":      round(kw_overlap, 3),
                "kw_hits":         kw_hits,
                "gold_description":     g.get("description"),
                "trace_summary":        (fa.get("summary") or "")[:300],
            }) + "\n")

    fail_fh.close()

    # 4) Write summary
    md_path = out_dir / "drugbank_crosscheck_summary.md"
    lines = []
    lines.append(f"# DrugBank cross-check  -- {len(traces):,} teacher_clean traces\n")
    lines.append(f"- Skipped (no gold row):     **{n_skip_no_gold:,}**")
    lines.append(f"- Skipped (parse fail):      **{n_no_parse:,}**")
    lines.append(f"- Skipped (abstention):      **{n_abstain:,}**")
    lines.append(f"- **Committed (audited):     {n_committed:,}**\n")

    lines.append("## Axis pass-rate per tier\n")
    lines.append("| Tier | Family | Subtype-kw | Polarity | Direction | All-3 |")
    lines.append("|---|---|---|---|---|---|")
    for tier in sorted(agg.keys()):
        row = [tier]
        for ax in ("family", "subtype_kw", "polarity", "direction"):
            d = agg[tier].get(ax, {"pass": 0, "fail": 0, "n": 0})
            n = d["n"]
            if n == 0:
                row.append("--")
            else:
                rate = d["pass"] / n
                row.append(f"{rate*100:.1f}% ({d['pass']}/{n})")
        # all-3 = family AND polarity AND direction
        f = agg[tier].get("family", {"pass": 0, "n": 0})
        p = agg[tier].get("polarity", {"pass": 0, "n": 0})
        dn = agg[tier].get("direction", {"pass": 0, "n": 0})
        # approximate all-3 from per-family records (we already wrote failures)
        # use fam_agg recomputed below; for tier breakdown, infer differently:
        row.append("--")  # filled in family-stratified table
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## Family-stratified all-3 pass rate\n")
    lines.append("| Family | n | all-3 pass | any-fail | rate |")
    lines.append("|---|---|---|---|---|")
    total_n = total_pass = 0
    for fam, d in sorted(fam_agg.items()):
        rate = (d["all_pass"] / d["n"]) if d["n"] else 0
        lines.append(f"| {fam} | {d['n']:,} | {d['all_pass']:,} | {d['any_fail']:,} | {rate*100:.1f}% |")
        total_n += d["n"]; total_pass += d["all_pass"]
    lines.append(f"| **TOTAL** | **{total_n:,}** | **{total_pass:,}** | **{total_n-total_pass:,}** | **{(total_pass/total_n*100) if total_n else 0:.1f}%** |")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"\n[crosscheck] summary -> {md_path}")
    print(f"[crosscheck] failures -> {fail_path}")
    print()
    print("\n".join(lines[-(8 + len(fam_agg)):]))


if __name__ == "__main__":
    main()
