"""Smoke tests for the Phase E metrics library.

These run fast with synthetic fixtures so the metrics can be exercised
before teacher traces land on disk.  Run with:
    python -m tests.test_metrics
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.mfs import mfs_trace, mfs_corpus                     # noqa: E402
from src.metrics.mps import MirrorRecord, mps_corpus                  # noqa: E402
from src.metrics.cfs import CounterfactualRecord, cfs_corpus          # noqa: E402
from src.metrics.au import AbstentionRecord, au_single, au_curve      # noqa: E402
from src.metrics.rpc import rpc_trace, rpc_corpus                     # noqa: E402
from src.metrics.slfs import slfs_trace, slfs_corpus                  # noqa: E402
from src.metrics.csa import CsaRecord, csa_corpus                     # noqa: E402
from src.metrics.hr import hr_trace, hr_corpus                        # noqa: E402
from src.metrics.ris import RisRecord, ris_corpus                     # noqa: E402


def _near(a, b, tol=1e-3):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- shared fixtures

CONTEXT_IDS = {'cyp3a4_sub', 'cyp2d6_inh', 'pathway_jaccard', 'hsa:05200'}


def _trace_committed():
    return {
        'steps': [
            {'step_id': 1, 'role': 'pk_flag',
             'claim': 'Drug A is metabolized by CYP3A4 and inhibits CYP3A4 activity.',
             'evidence_ids': ['cyp3a4_sub'], 'prm_score': 0.88},
            {'step_id': 2, 'role': 'pk_overlap',
             'claim': 'Metabolic clearance of drug B via CYP3A4 is reduced.',
             'evidence_ids': ['cyp3a4_sub', 'pathway_jaccard = 0.6'],
             'prm_score': 0.82},
            {'step_id': 3, 'role': 'conclusion',
             'claim': 'Metabolic interaction likely.',
             'evidence_ids': [], 'prm_score': 0.95},
        ],
        'final_answer': {'family': 'PK_Metabolism', 'direction_tag': 'a_to_b',
                         'abstain': False},
    }


def _trace_abstained():
    return {
        'steps': [
            {'step_id': 1, 'role': 'evidence_gap',
             'claim': 'Insufficient PK data in retrieved context.',
             'evidence_ids': [], 'prm_score': 0.70},
            {'step_id': 2, 'role': 'abstention',
             'claim': 'Cannot confidently commit to a mechanism family.',
             'evidence_ids': [], 'prm_score': 0.75},
        ],
        'final_answer': {'family': 'n/a', 'direction_tag': 'n/a', 'abstain': True},
    }


# ---------------------------------------------------------------- MFS
def test_mfs():
    assert _near(mfs_trace(_trace_committed(), CONTEXT_IDS), 1.0), "all-resolved -> 1.0"
    assert _near(mfs_trace(_trace_abstained(), CONTEXT_IDS), 0.0), "pure abstention -> 0 (no rationale steps)"
    rep = mfs_corpus([
        {'trace': _trace_committed(), 'context_ids': CONTEXT_IDS, 'family': 'PK_Metabolism'},
        {'trace': _trace_abstained(), 'context_ids': CONTEXT_IDS, 'family': 'AdverseRisk'},
    ])
    assert _near(rep['weighted_mfs'], 0.5), f"corpus weighted_mfs = {rep['weighted_mfs']}"
    assert rep['n_degenerate'] == 1
    print("[MFS]   PASS")


# ---------------------------------------------------------------- MPS
def test_mps():
    r_flip_ok = MirrorRecord('P2', 'PK_Metabolism', False, 'A',
                             {'family': 'PK_Metabolism', 'direction_tag': 'a_to_b'},
                             {'family': 'PK_Metabolism', 'direction_tag': 'b_to_a'})
    r_flip_fail = MirrorRecord('P3', 'PK_Metabolism', False, 'A',
                               {'family': 'PK_Metabolism', 'direction_tag': 'a_to_b'},
                               {'family': 'PK_Metabolism', 'direction_tag': 'a_to_b'})
    rep = mps_corpus([r_flip_ok, r_flip_fail])
    assert _near(rep['mps'], 0.5), f"mps = {rep['mps']}"
    assert rep['n_family_correct_both'] == 2
    print("[MPS]   PASS")


# ---------------------------------------------------------------- CfS
def test_cfs():
    rec_rel = CounterfactualRecord('P1', 'cyp3a4_inh', True,
        {'A': 0.9, 'B': 0.05, 'C': 0.05},
        {'A': 0.1, 'B': 0.45, 'C': 0.45})
    rec_null = CounterfactualRecord('P1', 'oct1_sub', False,
        {'A': 0.9, 'B': 0.05, 'C': 0.05},
        {'A': 0.89, 'B': 0.06, 'C': 0.05})
    rep = cfs_corpus([rec_rel, rec_null])
    assert rep['cfs_relevant'] > rep['cfs_null'] * 10, "relevant flip should dominate"
    assert rep['cfs_gap'] > 0
    print("[CfS]   PASS")


# ---------------------------------------------------------------- AU
def test_au():
    recs = [AbstentionRecord(f'P{i}', pred_correct=True, abstained=False) for i in range(81)]
    recs += [AbstentionRecord(f'P{i}', pred_correct=False, abstained=False) for i in range(81, 90)]
    recs += [AbstentionRecord(f'P{i}', pred_correct=False, abstained=True) for i in range(90, 100)]
    rep = au_single(recs, alpha=1.0)
    assert _near(rep['au'], 0.72), f"au = {rep['au']}"
    assert _near(rep['coverage'], 0.9)
    print("[AU]    PASS")


# ---------------------------------------------------------------- RPC
def test_rpc():
    tr_coherent = _trace_committed()
    tr_incoherent = {
        'steps': [
            {'step_id': 1, 'role': 'pk_flag',
             'claim': 'Drug A is metabolized by CYP3A4.',
             'evidence_ids': ['cyp3a4_sub']},
        ],
        'final_answer': {'family': 'AdverseRisk', 'abstain': False},
    }
    assert rpc_trace(tr_coherent, gold_family='PK_Metabolism').is_coherent is True
    assert rpc_trace(tr_incoherent, gold_family='AdverseRisk').is_coherent is False
    assert rpc_trace(_trace_abstained(), gold_family='AdverseRisk').is_coherent is True
    print("[RPC]   PASS")


# ---------------------------------------------------------------- SLFS
def test_slfs():
    t = _trace_committed()
    assert _near(slfs_trace(t), (0.88 + 0.82 + 0.95) / 3, 1e-2)
    excl = slfs_trace(t, exclude_meta=True)
    assert _near(excl, (0.88 + 0.82) / 2, 1e-2)
    print("[SLFS]  PASS")


# ---------------------------------------------------------------- CSA
def test_csa():
    recs = [
        CsaRecord('P1', 'PK_Metabolism', 'PK_Metabolism', 'PK_Metabolism'),      # both right
        CsaRecord('P2', 'PK_Metabolism', 'PK_Metabolism', 'AdverseRisk'),        # ab ok, ba wrong
        CsaRecord('P3', 'AdverseRisk',   'AdverseRisk',   'AdverseRisk'),        # both right
        CsaRecord('P4', 'PD_Synergy',    'PD_Antagonism', 'PD_Antagonism'),      # both wrong but consistent
    ]
    rep = csa_corpus(recs)
    assert _near(rep['csa'], 0.5), f"csa = {rep['csa']}"
    assert _near(rep['consistency'], 0.75), f"consistency = {rep['consistency']}"
    print("[CSA]   PASS")


# ---------------------------------------------------------------- HR
def test_hr():
    known = {'DB00543', 'CYP3A4', 'CYP2D6', 'P-gp', 'ABCB1'}
    t = {'steps': [
        {'step_id':1,'role':'pk_flag','claim':'Drug A (DB99999) inhibits CYP22Q4.'},
        {'step_id':2,'role':'pk_overlap','claim':'The CYP9Z7 pathway overlaps with OATP9X9.'},
    ]}
    r = hr_trace(t, known)
    assert r.n_entities == 4 and r.n_unknown == 4
    assert _near(r.hr, 1.0)

    t_good = _trace_committed()
    r2 = hr_trace(t_good, known)
    assert r2.n_unknown == 0, f"expected 0 unknowns, got {r2.unknown_entities}"
    print("[HR]    PASS")


# ---------------------------------------------------------------- RIS
def test_ris():
    recs = [
        RisRecord('P1', 'PK_Metabolism', 'PK_Metabolism', 'AdverseRisk',  'PK_Metabolism'),
        RisRecord('P2', 'PK_Metabolism', 'PK_Metabolism', 'AdverseRisk',  'PK_Metabolism'),
        RisRecord('P3', 'AdverseRisk',   'AdverseRisk',   'AdverseRisk',  'PD_Synergy'),
        RisRecord('P4', 'PK_Metabolism', 'AdverseRisk',   'AdverseRisk',  'PK_Metabolism'),
    ]
    rep = ris_corpus(recs)
    # acc_true = 3/4, acc_adv = 1/4, ris = 0.5
    assert _near(rep['acc_true_ev'], 0.75)
    assert _near(rep['acc_adv_ev'], 0.25)
    assert _near(rep['ris'], 0.5)
    # baseline acc_no_ev = 3/4 -> delta_true = 0, delta_adv = -0.5
    assert _near(rep['acc_no_ev'], 0.75)
    print("[RIS]   PASS")


# ---------------------------------------------------------------- Driver
if __name__ == "__main__":
    for t in [
        test_mfs, test_mps, test_cfs, test_au, test_rpc,
        test_slfs, test_csa, test_hr, test_ris,
    ]:
        t()
    print()
    print("All 9 metric tests passed.")
