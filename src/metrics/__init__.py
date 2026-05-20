"""DDI Verifier — metrics library.

Novel metrics for mechanism-faithful DDI evaluation :

    MFS   -- Mechanism-Faithfulness Score   (mfs.py)
    MPS   -- Mirror-Pair Separation          (mps.py)
    CfS   -- Counterfactual Sensitivity      (cfs.py)
    THS   -- Taxonomy Hierarchy Score        (ths.py -- pre-existing)
    AU    -- Abstention Utility              (au.py)
    RPC   -- Reasoning-Prediction Coherence  (rpc.py)
    SLFS  -- Step-Level Faithfulness Score   (slfs.py)
    CSA   -- Cross-Symmetry Agreement        (csa.py)
    HR    -- Hallucination Rate              (hr.py)
    RIS   -- Retrieval Influence Score       (ris.py)
    MOR   -- Mechanistic Overlap Rate        (mor.py — retrieval audit)

All scorers share the evidence-resolution convention from
`src.teacher.evidence_resolution` so metric reports stay consistent with
Phase B QC reports.
"""

from . import cfs, csa, hr, mfs, mps, rpc, ris, slfs, ths  # noqa: F401

# `mor` is an optional retrieval-audit metric (A7) that pulls in rdkit.
# Nothing in run_full_eval uses it, so keep its import lazy/optional so a
# missing rdkit on a GPU node does not block C8 evaluation.
try:
    from . import mor  # noqa: F401
except ImportError as _mor_err:  # pragma: no cover
    import warnings
    warnings.warn(
        f"src.metrics.mor not loaded ({_mor_err.__class__.__name__}: "
        f"{_mor_err}). Install rdkit if you need the MOR retrieval audit; "
        "run_full_eval does not need it.",
        stacklevel=2,
    )
