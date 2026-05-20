"""A5 — Hierarchical Mechanism-Aware Label Taxonomy .

Maps every DrugBank pair description into a 3-level label:

    Family   ∈ {PK_Absorption, PK_Distribution, PK_Metabolism, PK_Excretion,
                PD_Activity, Efficacy, AdverseRisk, Other}
    Subtype  : string (data-driven: the activity-adjective for PD, the event
               noun for AdverseRisk, the organ/route for PK, etc.).  Tail
               subtypes with <50 pairs bucket into `misc_<family>` —
               crucially NOT 'other', so family signal is preserved.
    Direction:
      subject_drugbank_id  — perpetrator (causes the effect)
      object_drugbank_id   — victim (whose PK/PD changes)
      bidirectional        — True for symmetric AdverseRisk frames
    Polarity: up | down | risk  (risk frames always 'increased')

The parser is deterministic — a cascade of regexes designed from the 928
normalized templates found in description corpus exploration. Every description must land in a named
Family (no silent 'other' collapse); only hand-drafted novel phrasings
remain in `Other`.

Outputs (data_processed/):
    labels_hierarchical.parquet   # one row per pair, with all fields above
    taxonomy_schema.json          # family/subtype vocabularies + counts
Audit (outputs/audit/):
    a5_taxonomy_report.md
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT / "data_processed" / "pairs.parquet"
DRUGS = ROOT / "data_processed" / "drugs.parquet"
BRANDS = ROOT / "data_processed" / "drug_brands.parquet"
OUT_LABELS = ROOT / "data_processed" / "labels_hierarchical.parquet"
OUT_SCHEMA = ROOT / "data_processed" / "taxonomy_schema.json"
OUT_DROPPED = ROOT / "outputs" / "audit" / "a5_dropped_pairs.parquet"
AUDIT_MD = ROOT / "outputs" / "audit" / "a5_taxonomy_report.md"

RARE_SUBTYPE_THRESHOLD = 50  # subtype with < N pairs collapses to misc_<family>

# Canonicalization map — redundant subtypes merged into the dominant form.
# Same clinical concept expressed with different phrasings in DrugBank descriptions.
SUBTYPE_CANONICAL: dict[tuple[str, str], str] = {
    # "The excretion of X..." and "X may decrease the excretion rate of Y..."
    # both describe renal excretion modulation — fold into the larger bin.
    ("PK_Excretion", "excretion"): "excretion_rate",
    # "The absorption of X..." ≡ "X can cause a decrease in the absorption of Y..."
    ("PK_Absorption", "absorption"): "absorption_change",
}

# ── Ordered family patterns ──────────────────────────────────────────────────
# Each entry: (family, compiled regex, callable taking match → (subtype, polarity,
#   subject_tag, object_tag)) where subject_tag/object_tag ∈ {'A','B','both'}.
# The tags map to drugbank_ids using raw_subject_id semantics in apply().

def _norm_subtype(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unspecified"


# Precompile all patterns.
PATTERNS: list[tuple[str, re.Pattern, callable]] = []

def add(family: str, pattern: str, handler):
    PATTERNS.append((family, re.compile(pattern), handler))


# 1. AdverseRisk — always symmetric (both drugs are "combined"), polarity=risk.
#    "The risk or severity of <EVENT> can be increased when <X> is combined with <Y>."
add("AdverseRisk",
    r"^The risk or severity of (?P<event>.+?) can be increased when <(?P<x>A|B)> is combined with <(?P<y>A|B)>\.$",
    lambda m: (_norm_subtype(m.group("event")), "risk", "both", "both"))

# 1b. AdverseRisk (decrease variant) — same structure, "decreased" polarity.
#     Semantically still a risk/adverse frame (protective interaction).
add("AdverseRisk",
    r"^The risk or severity of (?P<event>.+?) can be decreased when <(?P<x>A|B)> is combined with <(?P<y>A|B)>\.$",
    lambda m: (_norm_subtype(m.group("event")), "risk_down", "both", "both"))

# 2. PK_Metabolism — "The metabolism of <D> can be {increased|decreased} when combined with <Y>."
#    Here <D> is the OBJECT (metabolism of D is affected); <Y> is the SUBJECT (modulator).
add("PK_Metabolism",
    r"^The metabolism of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when combined with <(?P<subj>A|B)>\.$",
    lambda m: ("metabolism", "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 3. PK_Distribution — "The serum concentration of <D> can be {increased|decreased} when it is combined with <Y>."
add("PK_Distribution",
    r"^The serum concentration of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when it is combined with <(?P<subj>A|B)>\.$",
    lambda m: ("serum_concentration", "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 3b. PK_Distribution — "The serum concentration of the active metabolites of <D> can be..."
add("PK_Distribution",
    r"^The serum concentration of the active metabolites? of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when (it is )?(used|combined) (in combination )?with <(?P<subj>A|B)>\.$",
    lambda m: ("active_metabolite_serum_conc",
               "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 4. PK_Absorption — "<X> can cause a/an {decrease|increase} in the absorption of <Y> resulting in..."
add("PK_Absorption",
    r"^<(?P<subj>A|B)> can cause a(?:n)? (?P<pol>decrease|increase) in the absorption of <(?P<obj>A|B)>.*$",
    lambda m: ("absorption_change",
               "down" if m.group("pol") == "decrease" else "up",
               m.group("subj"), m.group("obj")))

# 4b. PK_Absorption — "The {bioavailability|absorption} of <X> can be {increased|decreased} when combined with <Y>."
add("PK_Absorption",
    r"^The (?P<what>bioavailability|absorption) of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when combined with <(?P<subj>A|B)>\.$",
    lambda m: (m.group("what"), "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 5. PK_Excretion — "<X> may {decrease|increase} the excretion rate of <Y>..."
add("PK_Excretion",
    r"^<(?P<subj>A|B)> may (?P<pol>decrease|increase) the excretion rate of <(?P<obj>A|B)>.*$",
    lambda m: ("excretion_rate",
               "down" if m.group("pol") == "decrease" else "up",
               m.group("subj"), m.group("obj")))

# 5b. PK_Excretion — "The excretion of <X> can be {increased|decreased} when combined with <Y>."
add("PK_Excretion",
    r"^The excretion of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when combined with <(?P<subj>A|B)>\.$",
    lambda m: ("excretion", "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 5c. PK_Excretion — "The renal clearance of <X> can be {decreased|increased} when combined with <Y>."
add("PK_Excretion",
    r"^The renal clearance of <(?P<obj>A|B)> can be (?P<pol>decreased|increased) when combined with <(?P<subj>A|B)>\.$",
    lambda m: ("renal_clearance", "down" if m.group("pol") == "decreased" else "up",
               m.group("subj"), m.group("obj")))

# 5d. PK_Distribution — "The protein binding of <X> can be {decreased|increased} when combined with <Y>."
add("PK_Distribution",
    r"^The protein binding of <(?P<obj>A|B)> can be (?P<pol>decreased|increased) when combined with <(?P<subj>A|B)>\.$",
    lambda m: ("protein_binding", "down" if m.group("pol") == "decreased" else "up",
               m.group("subj"), m.group("obj")))

# 6. PD_Activity — "<X> may {increase|decrease} the <ADJ> activities of <Y>."
#    (The most productive PD frame, covers thousands of subtypes.)
add("PD_Activity",
    r"^<(?P<subj>A|B)> may (?P<pol>increase|decrease) the (?P<adj>.+?) activities of <(?P<obj>A|B)>\.$",
    lambda m: (_norm_subtype(m.group("adj")),
               "up" if m.group("pol") == "increase" else "down",
               m.group("subj"), m.group("obj")))

# 6b. PD_Activity — variant "<X> may {increase|decrease} the <X>ic/<X>al <ADJ> activities..."
# (already covered by 6; the ADJ group is non-greedy so it matches.)

# 7. Efficacy — "The therapeutic efficacy of <X> can be {increased|decreased} when used in combination with <Y>."
add("Efficacy",
    r"^The therapeutic efficacy of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when used in combination with <(?P<subj>A|B)>\.$",
    lambda m: ("therapeutic_efficacy",
               "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 7b. Efficacy — diagnostic agents.
add("Efficacy",
    r"^<(?P<subj>A|B)> may (?P<pol>decrease|increase) effectiveness of <(?P<obj>A|B)> as a diagnostic agent\.$",
    lambda m: ("diagnostic_effectiveness",
               "down" if m.group("pol") == "decrease" else "up",
               m.group("subj"), m.group("obj")))

# 8. AdverseRisk — hypersensitivity reaction variant.
# "The risk of a hypersensitivity reaction to <X> is increased when it is combined with <Y>."
# X = object (receives reaction); Y = subject (causes heightened reaction)
add("AdverseRisk",
    r"^The risk of a hypersensitivity reaction to <(?P<obj>A|B)> is increased when it is combined with <(?P<subj>A|B)>\.$",
    lambda m: ("hypersensitivity_reaction", "risk",
               m.group("subj"), m.group("obj")))

# 3c. PK_Distribution — active-metabolites with placeholder in middle clause.
# "The serum concentration of the active metabolites of <A> can be {increased|decreased} when <A> is used in combination with <B>."
add("PK_Distribution",
    r"^The serum concentration of the active metabolites? of <(?P<obj>A|B)> can be (?P<pol>increased|decreased) when <(?P<mid>A|B)> is used in combination with <(?P<subj>A|B)>\.$",
    lambda m: ("active_metabolite_serum_conc",
               "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))

# 3d. PK_Distribution — "reduced" variant with efficacy-loss trailing clause.
# "The serum concentration of the active metabolites of <A> can be reduced when <A> is used in combination with <B> resulting in a loss in efficacy."
add("PK_Distribution",
    r"^The serum concentration of the active metabolites? of <(?P<obj>A|B)> can be reduced when <(?P<mid>A|B)> is used in combination with <(?P<subj>A|B)>.*$",
    lambda m: ("active_metabolite_serum_conc", "down",
               m.group("subj"), m.group("obj")))

# 3e. PK_Distribution — literal metabolite name (not A/B) in "X, an active metabolite of A, can be ..."
# "The serum concentration of dextroamphetamine, an active metabolite of <A>, can be increased when used in combination with <B>."
add("PK_Distribution",
    r"^The serum concentration of [A-Za-z0-9\- ]+, an active metabolite of <(?P<obj>A|B)>, can be (?P<pol>increased|decreased|reduced) when used in combination with <(?P<subj>A|B)>\.$",
    lambda m: ("active_metabolite_serum_conc",
               "up" if m.group("pol") == "increased" else "down",
               m.group("subj"), m.group("obj")))


# ── Name substitution with synonyms + international brand names ─────────────
def build_name_table() -> dict[str, list[str]]:
    """drug_id → sorted list of aliases (longest first).

    Uses primary <name> + <synonyms> from drugs.parquet PLUS **international
    brand names** from drug_brands.parquet (brand_kind == 'international').
    Product-kind brands are excluded because popular drugs have 10k+ product
    entries (one per dosage/strength/country), which blew up O(aliases × text)
    in the earlier version.  International brands are the form actually used
    inside DrugBank DDI descriptions (e.g. "Mebicar" is the international
    brand of DB13522/Temgicoluril and appears in 524 descriptions).

    Longest-first ordering is crucial because many aliases are substrings of
    others (replace "Warfarin sodium" before "Warfarin").
    """
    print("[A5] building drug-name table (primary + synonyms + international brands) ...")
    drugs_tbl = pq.read_table(DRUGS, columns=["drugbank_id", "name", "synonyms"]).to_pylist()
    table: dict[str, set[str]] = {}
    for d in drugs_tbl:
        did = d["drugbank_id"]
        s: set[str] = set()
        if d.get("name"):
            s.add(d["name"].strip())
        for syn in (d.get("synonyms") or []):
            if syn and syn.strip():
                s.add(syn.strip())
        table[did] = s

    # Merge international brands into the same table
    try:
        brands_tbl = pq.read_table(
            BRANDS, columns=["drugbank_id", "brand_kind", "name"]
        ).to_pylist()
        added = 0
        for b in brands_tbl:
            if b.get("brand_kind") != "international":
                continue
            nm = b.get("name")
            if not nm or not nm.strip():
                continue
            did = b.get("drugbank_id")
            if did in table:
                before = len(table[did])
                table[did].add(nm.strip())
                if len(table[did]) > before:
                    added += 1
        print(f"[A5]   merged {added:,} international-brand aliases")
    except FileNotFoundError:
        print("[A5]   drug_brands.parquet not found — skipping brand aliases")

    return {did: sorted(s, key=len, reverse=True) for did, s in table.items()}


# Global table populated in main()
_NAME_TABLE: dict[str, list[str]] = {}
_ALIAS_RX_CACHE: dict[str, re.Pattern] = {}


def _alias_rx(alias: str) -> re.Pattern:
    rx = _ALIAS_RX_CACHE.get(alias)
    if rx is None:
        rx = re.compile(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])")
        _ALIAS_RX_CACHE[alias] = rx
    return rx


def normalize_description(a_name: str, b_name: str, text: str,
                          raw_subject_id: str, a_id: str, b_id: str) -> str:
    """Replace drug names with <A>/<B>.

    Convention: <A> = the drug that was the XML <drug> PARENT when the description
    was written (i.e. raw_subject_id). <B> = the other drug.
    For each pair we use ALL known aliases (primary name + synonyms + brands)
    of both drugs, longest-first. This fixes the common case where DrugBank
    descriptions use a synonym that doesn't match the primary <name> field.
    """
    subj_id = raw_subject_id
    obj_id = b_id if raw_subject_id == a_id else a_id

    subj_aliases = list(_NAME_TABLE.get(subj_id, []))
    obj_aliases = list(_NAME_TABLE.get(obj_id, []))
    # Fall back to the name on the pair row if the table didn't cover the drug
    if a_name and a_name not in (_NAME_TABLE.get(a_id) or []):
        (subj_aliases if subj_id == a_id else obj_aliases).append(a_name)
    if b_name and b_name not in (_NAME_TABLE.get(b_id) or []):
        (obj_aliases if subj_id == a_id else subj_aliases).append(b_name)

    # Interleave by length (longest across both sets first), breaking ties by
    # subject-first to avoid ambiguity when two drugs share an alias.
    all_aliases = sorted(
        [(a, "A") for a in subj_aliases] + [(a, "B") for a in obj_aliases],
        key=lambda pair: (-len(pair[0]), pair[1] != "A"),
    )

    out = text
    for alias, tag in all_aliases:
        if not alias:
            continue
        out = _alias_rx(alias).sub(f"<{tag}>", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def resolve_tag_to_id(tag: str, raw_subject_id: str, a_id: str, b_id: str) -> str | None:
    """tag ∈ {'A','B'} → drugbank_id. <A> was the raw_subject_id drug."""
    other_id = b_id if raw_subject_id == a_id else a_id
    if tag == "A":
        return raw_subject_id
    elif tag == "B":
        return other_id
    return None


def classify(template: str, raw_subject_id: str, a_id: str, b_id: str) -> dict:
    for family, rx, handler in PATTERNS:
        m = rx.match(template)
        if m is None:
            continue
        subtype, polarity, subj_tag, obj_tag = handler(m)
        if subj_tag == "both":
            return {
                "family": family, "subtype": subtype, "polarity": polarity,
                "subject_drugbank_id": None, "object_drugbank_id": None,
                "bidirectional": True,
            }
        return {
            "family": family, "subtype": subtype, "polarity": polarity,
            "subject_drugbank_id": resolve_tag_to_id(subj_tag, raw_subject_id, a_id, b_id),
            "object_drugbank_id":  resolve_tag_to_id(obj_tag,  raw_subject_id, a_id, b_id),
            "bidirectional": False,
        }
    # Unmatched: these pairs have a codename / investigational-drug name in the
    # description that isn't in our alias table (drugbank name + synonyms +
    # international brands).  Dropped from labels_hierarchical.parquet and
    # written to outputs/audit/a5_dropped_pairs.parquet for transparency.
    # Including them in training would pollute the teacher/student signal with
    # "unknown" family — better to exclude now than reason over weak labels.
    return {"family": "Other", "subtype": "unmatched", "polarity": None,
            "subject_drugbank_id": None, "object_drugbank_id": None,
            "bidirectional": False}


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    global _NAME_TABLE
    _NAME_TABLE = build_name_table()
    sample_alias_counts = Counter(len(v) for v in _NAME_TABLE.values())
    print(f"[A5] name table: {len(_NAME_TABLE):,} drugs; "
          f"alias-count distribution: {dict(sorted(sample_alias_counts.items()))}")

    print("[A5] loading pairs.parquet ...", flush=True)
    pairs = pq.read_table(PAIRS).to_pylist()
    print(f"[A5] {len(pairs):,} pairs", flush=True)

    rows: list[dict] = []
    fam_counter: Counter = Counter()
    subtype_counter: dict[str, Counter] = defaultdict(Counter)
    unmatched_samples: list[str] = []

    for i, p in enumerate(pairs):
        tmpl = normalize_description(
            p["a_name"] or "", p["b_name"] or "",
            p["description"] or "", p["raw_subject_id"], p["a_id"], p["b_id"])
        lab = classify(tmpl, p["raw_subject_id"], p["a_id"], p["b_id"])
        fam_counter[lab["family"]] += 1
        subtype_counter[lab["family"]][lab["subtype"]] += 1
        if lab["family"] == "Other" and len(unmatched_samples) < 40:
            unmatched_samples.append(tmpl)
        rows.append({
            "pair_id": p["pair_id"],
            "a_id": p["a_id"], "b_id": p["b_id"],
            "description": p["description"],
            "template": tmpl,
            "family": lab["family"],
            "subtype": lab["subtype"],
            "polarity": lab["polarity"],
            "subject_drugbank_id": lab["subject_drugbank_id"],
            "object_drugbank_id":  lab["object_drugbank_id"],
            "bidirectional": lab["bidirectional"],
        })
        if (i + 1) % 250_000 == 0:
            print(f"  processed {i+1:,} ({time.time()-t0:.0f}s)", flush=True)

    # ── Canonicalize redundant subtypes (phrasing variants → dominant form) ───
    for r in rows:
        key = (r["family"], r["subtype"])
        if key in SUBTYPE_CANONICAL:
            r["subtype"] = SUBTYPE_CANONICAL[key]
    # recompute subtype counters after canonicalization (for tail-collapse)
    subtype_counter = defaultdict(Counter)
    for r in rows:
        subtype_counter[r["family"]][r["subtype"]] += 1

    # ── Collapse rare subtypes to misc_<family> (NOT 'other') ─────────────────
    subtype_map: dict[tuple[str, str], str] = {}
    for family, counter in subtype_counter.items():
        for sub, n in counter.items():
            if n < RARE_SUBTYPE_THRESHOLD and family != "Other":
                subtype_map[(family, sub)] = f"misc_{family.lower()}"
    for r in rows:
        mapped = subtype_map.get((r["family"], r["subtype"]))
        if mapped:
            r["subtype_original"] = r["subtype"]
            r["subtype"] = mapped
        else:
            r["subtype_original"] = r["subtype"]

    # Recompute after collapse
    fam_counter2 = Counter(r["family"] for r in rows)
    subtype_counter2: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        subtype_counter2[r["family"]][r["subtype"]] += 1

    # ── Split kept vs dropped ────────────────────────────────────────────────
    # DROP: (a) "Other" — codename/investigational drugs that didn't resolve;
    #       (b) subject==object on directional rows — DrugBank cross-drug
    #           synonym anomaly (e.g. "Simendan" is a synonym of both DB00922
    #           and DB12286, making a directional label unresolvable).
    def _drop_reason(r):
        if r["family"] == "Other":
            return "unmatched_codename"
        if (not r["bidirectional"]) and r["subject_drugbank_id"] == r["object_drugbank_id"] \
                and r["subject_drugbank_id"]:
            return "ambiguous_synonym_same_id"
        return None
    for r in rows:
        r["drop_reason"] = _drop_reason(r)
    kept_rows = [r for r in rows if not r["drop_reason"]]
    dropped_rows = [r for r in rows if r["drop_reason"]]
    # strip drop_reason from kept (not needed downstream)
    for r in kept_rows:
        r.pop("drop_reason", None)
    if dropped_rows:
        pq.write_table(pa.Table.from_pylist(dropped_rows), OUT_DROPPED, compression="snappy")
        print(f"[A5] wrote {OUT_DROPPED.relative_to(ROOT)} ({len(dropped_rows):,} unmatched pairs)", flush=True)

    # ── Write labels parquet (kept rows only) ─────────────────────────────────
    pq.write_table(pa.Table.from_pylist(kept_rows), OUT_LABELS, compression="snappy")
    print(f"[A5] wrote {OUT_LABELS.relative_to(ROOT)} ({len(kept_rows):,} rows)", flush=True)

    # Reassign `rows` to kept-only for downstream counters/audit
    rows = kept_rows

    # Schema JSON
    schema = {
        "families": {f: fam_counter2[f] for f in sorted(fam_counter2)},
        "subtypes": {f: dict(subtype_counter2[f].most_common()) for f in sorted(subtype_counter2)},
        "rare_subtype_threshold": RARE_SUBTYPE_THRESHOLD,
        "n_rare_subtypes_collapsed": len(subtype_map),
    }
    OUT_SCHEMA.write_text(json.dumps(schema, indent=2, ensure_ascii=False))

    # ── Audit report ──────────────────────────────────────────────────────────
    n_kept = len(rows)
    n_dropped = len(dropped_rows)
    n_total_pairs = n_kept + n_dropped
    coverage = 100 * n_kept / n_total_pairs

    md = [
        "# A5 — Hierarchical taxonomy report\n",
        f"- Total input pairs: **{n_total_pairs:,}**",
        f"- Classified & kept: **{n_kept:,}** ({coverage:.2f}%)",
        f"- Dropped (unmatched codename drugs): **{n_dropped:,}** ({100*n_dropped/n_total_pairs:.2f}%) "
        f"→ written to `outputs/audit/a5_dropped_pairs.parquet` for transparency",
        f"- Rare-subtype threshold: <{RARE_SUBTYPE_THRESHOLD} pairs → `misc_<family>` (family preserved)",
        f"- Subtypes canonicalized (redundant phrasings merged): {len(SUBTYPE_CANONICAL)}",
        f"- Subtypes collapsed to misc_<family>: **{len(subtype_map)}**",
        f"- Build time: {time.time()-t0:.0f}s",
        "",
        "## Family distribution (kept rows only)",
        "",
        "| Family | Pairs | % |",
        "|---|---:|---:|",
    ]
    for fam, n in fam_counter2.most_common():
        if fam == "Other":
            continue
        md.append(f"| `{fam}` | {n:,} | {100*n/n_kept:.2f}% |")
    md.append("")
    n_families_kept = sum(1 for f in fam_counter2 if f != "Other")
    md.append(f"**Families: {n_families_kept}** (plan target ≤20 ✓).")

    # Subtype summary per family
    md.append("")
    md.append("## Subtype counts per family")
    md.append("")
    for fam in sorted(subtype_counter2):
        if fam == "Other":
            continue
        subs = subtype_counter2[fam]
        md.append(f"### `{fam}` — {len(subs)} distinct subtypes, {sum(subs.values()):,} pairs")
        md.append("")
        md.append("| Subtype | Pairs | % of family |")
        md.append("|---|---:|---:|")
        fam_total = sum(subs.values())
        for sub, n in subs.most_common(15):
            md.append(f"| `{sub}` | {n:,} | {100*n/fam_total:.1f}% |")
        if len(subs) > 15:
            md.append(f"| _(… {len(subs)-15} more)_ | | |")
        md.append("")
    # Unmatched samples
    if unmatched_samples:
        md.append("## Sample of unmatched ('Other') templates")
        md.append("")
        for tmpl in unmatched_samples:
            md.append(f"- `{tmpl}`")
        md.append("")

    # Direction coverage
    directional = sum(1 for r in rows if not r["bidirectional"])
    symmetric = n_kept - directional
    md.append("## Direction")
    md.append("")
    md.append(f"- Directional (subject→object resolved): **{directional:,}** "
              f"({100*directional/n_kept:.2f}%)")
    md.append(f"- Bidirectional / symmetric frames: **{symmetric:,}** "
              f"({100*symmetric/n_kept:.2f}%)")
    pol = Counter(r["polarity"] for r in rows)
    md.append(f"- Polarity counts: {dict(pol)}")
    md.append("")
    md.append("## Gates")
    md.append(f"- `coverage ≥ 99%`: **{'PASS' if coverage >= 99 else 'FAIL'}** "
              f"(actual {coverage:.2f}%)")
    md.append(f"- `len(families) ≤ 20`: **{'PASS' if n_families_kept <= 20 else 'FAIL'}**")
    md.append(f"- `no subtype == 'other'`: "
              f"**{'PASS' if 'other' not in [s for f, d in subtype_counter2.items() if f != 'Other' for s in d] else 'FAIL'}**")
    md.append(f"- `no 'Other' family in output`: "
              f"**{'PASS' if 'Other' not in [r['family'] for r in rows] else 'FAIL'}** "
              f"(Other pairs moved to `a5_dropped_pairs.parquet`)")

    AUDIT_MD.write_text("\n".join(md) + "\n")
    print(f"[A5] wrote {AUDIT_MD.relative_to(ROOT)}")
    print(f"[A5] families: {dict(fam_counter2)}")
    print(f"[A5] coverage: {coverage:.2f}%")


if __name__ == "__main__":
    main()
