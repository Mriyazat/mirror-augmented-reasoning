#!/bin/bash
# ============================================================================
# Source this script to activate the DDI Verifier environment on a SLURM
# H100 GPU node. This is an example for a typical 4 x H100 80GB node and
# may require minor tweaks for your specific cluster.
#
# Example usage (inside your allocation):
#   salloc --time=8:00:00 --cpus-per-task=64 --mem=0 \
#          --gres=gpu:h100:4 --account=${SLURM_ACCOUNT:-your_account}
#   cd /path/to/EMNLP2026_DDI_Verifier_Release
#   source scripts/cluster_examples/activate_env.sh
#
# First-time setup only:
#   bash scripts/cluster_examples/setup_env.sh
# ============================================================================

# ---------- cluster modules (adapt to your cluster's module names) ----------
# Note: on some HPC sites RDKit must be loaded BEFORE the venv is
# activated because the cluster ships a placeholder "noinstall" wheel.
module load StdEnv/2023 2>/dev/null || true
module load gcc/12.3 python/3.11 cuda/12.2 arrow opencv/4.12.0 rdkit/2024.09.6 2>/dev/null || true

# ---------- virtual env ----------
# Override at runtime with: DDI_VENV=/path/to/venv source activate_env.sh
DDI_VENV="${DDI_VENV:-$HOME/ddi_venv}"
if [ ! -d "$DDI_VENV" ]; then
    echo "[activate_env] venv not found at $DDI_VENV"
    echo "[activate_env] Set DDI_VENV=... or run scripts/cluster_examples/setup_env.sh"
    return 1 2>/dev/null || exit 1
fi
source "$DDI_VENV/bin/activate"

# ---------- PYTHONPATH so `python -m src.*` works from repo root ----------
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ---------- HF cache on scratch (pre-download on login node, offline at run time) --
export SCRATCH="${SCRATCH:-$HOME/scratch}"
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
# Block any attempt to hit the HF hub from a compute node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------- PyTorch / CUDA memory tuning for H100 80GB ----------
# expandable_segments reduces fragmentation in long-running vLLM / FSDP jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------- NCCL tuning for 4 x H100 multi-GPU ----------
export NCCL_DEBUG=WARN
# Configure IB_HCA / SOCKET overrides here if your cluster requires them.

# ---------- vLLM / FSDP / Accelerate ----------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# ---------- project paths (large outputs/checkpoints live under scratch) ----
export DDI_ROOT="$(pwd)"
export DDI_OUTPUTS="${DDI_OUTPUTS:-$SCRATCH/ddi_outputs}"
export DDI_CKPT="${DDI_CKPT:-$SCRATCH/ddi_checkpoints}"
mkdir -p "$DDI_OUTPUTS" "$DDI_CKPT"

# ---------- status banner ----------
echo "============================================================"
echo "[activate_env] DDI Verifier environment ready"
echo "  Venv:          $DDI_VENV"
echo "  Python:        $(python --version 2>&1)"
echo "  PyTorch:       $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'n/a')"
echo "  CUDA avail:    $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'n/a')"
echo "  GPUs:          $(python -c 'import torch; print(torch.cuda.device_count(), \"x\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")' 2>/dev/null || echo 'no GPU visible')"
echo "  RDKit:         $(python -c 'import rdkit; print(rdkit.__version__)' 2>/dev/null || echo 'MISSING - module load rdkit/<ver> before venv')"
echo "  HF cache:      $HF_HOME"
echo "  Outputs:       $DDI_OUTPUTS"
echo "  Checkpoints:   $DDI_CKPT"
echo ""
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  (1 = blocked, pre-download models on a login node)"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "  NCCL_DEBUG=$NCCL_DEBUG"
echo "============================================================"
