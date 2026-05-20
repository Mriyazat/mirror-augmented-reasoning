#!/bin/bash
# ============================================================================
# One-time setup for the DDI Verifier on a SLURM cluster (4 x H100 node).
# This is an example; adapt module names and venv paths to your site.
#
# STEP 0 (login node OR small allocation):
#   salloc --time=1:00:00 --cpus-per-task=4 --mem=16G \
#          --account=${SLURM_ACCOUNT:-your_account}
#
# STEP 1 (this script):
#   bash scripts/cluster_examples/setup_env.sh
#
# STEP 2 (future H100 sessions, just activate):
#   salloc --time=8:00:00 --cpus-per-task=32 --mem=256G \
#          --gres=gpu:h100:4 --account=${SLURM_ACCOUNT:-your_account}
#   source scripts/cluster_examples/activate_env.sh
# ============================================================================
set -e

echo "============================================================"
echo "[setup] DDI Verifier environment setup"
echo "============================================================"

# Cluster modules — adapt to your site's module naming.
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 python/3.11 cuda/12.2 arrow opencv/4.12.0 2>/dev/null || true

DDI_VENV="${DDI_VENV:-$HOME/ddi_venv}"

if [ -d "$DDI_VENV" ]; then
    echo "[setup] Reusing existing venv: $DDI_VENV"
    source "$DDI_VENV/bin/activate"
else
    echo "[setup] Creating fresh venv: $DDI_VENV"
    virtualenv --no-download "$DDI_VENV" 2>/dev/null || python -m venv "$DDI_VENV"
    source "$DDI_VENV/bin/activate"
    pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip
fi

echo "[setup] Installing dependencies..."

# ----- data stack -----
pip install --no-index pyarrow numpy pandas scipy scikit-learn 2>/dev/null || \
    pip install pyarrow numpy pandas scipy scikit-learn
pip install --no-index rdkit 2>/dev/null || pip install 'rdkit>=2024.3' 2>/dev/null || true
pip install --no-index xgboost 2>/dev/null || pip install xgboost
pip install --no-index matplotlib seaborn tqdm pyyaml jsonlines lxml requests 2>/dev/null || \
    pip install matplotlib seaborn tqdm pyyaml jsonlines lxml requests

# ----- LLM training/inference stack -----
# vLLM for teacher inference (e.g. 70B-class model with tensor-parallel on 4 x H100).
pip install --no-index vllm 2>/dev/null || echo "[setup] vllm: install manually on a login node with internet access"
pip install --no-index torch 2>/dev/null || echo "[setup] torch: using cluster default"
# HuggingFace ecosystem for student SFT/DPO.
pip install --no-index transformers accelerate peft bitsandbytes 2>/dev/null || \
    pip install transformers accelerate peft bitsandbytes
pip install --no-index datasets evaluate safetensors 2>/dev/null || \
    pip install datasets evaluate safetensors
# Preference-learning library (DPO, SFT).
pip install trl 2>/dev/null || echo "[setup] trl: install on a login node with internet access"

# ----- eval / scoring -----
pip install rouge-score bert-score 2>/dev/null || echo "[setup] rouge/bert-score: optional"

# ----- paths -----
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export HF_HOME="$SCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME" "$SCRATCH/ddi_outputs" "$SCRATCH/ddi_checkpoints"

echo ""
echo "============================================================"
echo "[setup] DONE."
echo ""
echo "Verify:"
echo "  source scripts/cluster_examples/activate_env.sh"
echo "  python -c 'import torch, vllm, transformers, peft, trl; print(\"all good\")'"
echo ""
echo "Next: pre-download models on a login node (internet available):"
echo "  HF_HUB_OFFLINE=0 python scripts/cluster_examples/download_models.py"
echo "============================================================"
