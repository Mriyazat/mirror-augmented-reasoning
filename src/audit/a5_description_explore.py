"""A5.0 — Description corpus exploration BEFORE writing the taxonomy.

We normalize each description (replace drug names with <A>/<B> placeholders),
then count exact templates. This shows the real structural vocabulary we must cover.

Writes:
  outputs/audit/a5_top_templates.tsv  — count, template
  outputs/audit/a5_top_templates_preview.md
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "data_processed" / "pairs.parquet"
DRUGS = ROOT / "data_processed" / "drugs.parquet"
OUT_TSV = ROOT / "outputs" / "audit" / "a5_top_templates.tsv"
OUT_MD  = ROOT / "outputs" / "audit" / "a5_top_templates_preview.md"


def build_namewise_placeholder_replacer(pairs_rows) -> callable:
    """We replace only the two drug names present in the pair's own row
    (not a global name table, which would be slow and error-prone)."""
    def normalize(a_name: str, b_name: str, text: str, raw_subject_id: str,
                  a_id: str, b_id: str) -> str:
        # Tag: X = the 'subject' drug of the sentence, Y = the co-mentioned drug.
        # (raw_subject_id corresponds to whichever of {a_id, b_id} was the DrugBank XML
        # <drug> parent when this interaction was written.)
        subj_name = a_name if raw_subject_id == a_id else b_name
        obj_name  = b_name if raw_subject_id == a_id else a_name
        # Replace longer first so substrings don't collide.
        names = sorted(filter(None, [subj_name, obj_name]), key=len, reverse=True)
        out = text
        for n in names:
            if n == subj_name:
                out = out.replace(n, "<A>")
            else:
                out = out.replace(n, "<B>")
        return out
    return normalize


def main():
    print("[A5.0] reading pairs.parquet + drugs.parquet...")
    pairs = pq.read_table(PAIRS, columns=[
        "a_id", "b_id", "a_name", "b_name", "description", "raw_subject_id"
    ]).to_pylist()
    print(f"[A5.0] {len(pairs):,} pairs loaded")

    normalize = build_namewise_placeholder_replacer(pairs)

    tpl = Counter()
    for p in pairs:
        norm = normalize(p["a_name"] or "", p["b_name"] or "",
                         p["description"] or "",
                         p["raw_subject_id"], p["a_id"], p["b_id"])
        # collapse whitespace
        norm = re.sub(r"\s+", " ", norm).strip()
        tpl[norm] += 1

    total = sum(tpl.values())
    print(f"[A5.0] {len(tpl):,} unique normalized templates (cover {total:,} pairs)")

    cum = 0
    top_n_cover = None
    sorted_items = tpl.most_common()
    for i, (_, c) in enumerate(sorted_items, 1):
        cum += c
        if cum / total >= 0.95 and top_n_cover is None:
            top_n_cover = i
            break

    # Write full TSV
    with OUT_TSV.open("w") as f:
        f.write("count\ttemplate\n")
        for t, c in sorted_items:
            f.write(f"{c}\t{t}\n")

    # Preview MD (top 60 templates + coverage summary)
    md = [
        "# A5.0 — DrugBank description templates\n",
        f"- Total pair rows: **{total:,}**",
        f"- Unique normalized templates: **{len(tpl):,}**",
        f"- Templates needed to cover **95%** of rows: **{top_n_cover:,}**",
        "",
        "## Top 60 templates (frequency-sorted)",
        "",
        "| Rank | Count | Template |",
        "|---:|---:|---|",
    ]
    for i, (t, c) in enumerate(sorted_items[:60], 1):
        tt = t.replace("|", "\\|")
        if len(tt) > 220:
            tt = tt[:217] + "…"
        md.append(f"| {i} | {c:,} | {tt} |")
    md.append("")
    md.append(f"Full list in `{OUT_TSV.relative_to(ROOT)}`.")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[A5.0] wrote {OUT_MD.relative_to(ROOT)} + {OUT_TSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
