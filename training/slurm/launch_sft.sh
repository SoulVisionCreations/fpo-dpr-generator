#!/bin/bash
#SBATCH --job-name=dpr-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# LoRA SFT of the DPR narrative student (Option B: author whole DPR with
# {{TABLE:id}} placeholders; engine injects exact tables + verifies).
#
#   sbatch launch_sft.sh                              # gpt-oss-20b, 2 epochs
#   sbatch launch_sft.sh google/gemma-3-4b-it 3       # small student, 3 epochs
#
# Arg 1 = model id (default openai/gpt-oss-20b), Arg 2 = epochs (default 2).
# Data: data/finetune/train.jsonl + val.jsonl (from teacher.py --collect)
# Output: models/<model-slug>[_tag]/adapter/   Logs: logs/slurm_%j.{out,err}
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${1:-openai/gpt-oss-20b}"
EPOCHS="${2:-2}"
TAG="${3:-}"                                    # optional version tag (e.g. v2 after adding data)
# per-model output dir so different students never overwrite each other
SLUG=$(echo "$MODEL" | tr '/:' '__')
OUT="models/${SLUG}${TAG:+_$TAG}"

cd "$(dirname "$0")/../.." || exit 1          # repo root
mkdir -p logs "$OUT"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Use every visible GPU together; sft_train.py shards the model weights across the ones
# with real free room, so each keeps runtime headroom for the per-sample
# loss/attention spikes. (MXFP4/gpt-oss falls back to a single GPU inside the
# script, since its dequant can't shard.)
# Order them roomiest-first so cuda:0 (where accelerate materialises the logits
# + loss) is the GPU with the most free memory. Cap the range with DPR_MAX_GPU_INDEX.
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' -v max="${DPR_MAX_GPU_INDEX:-99}" '$1+0<=max {print $2+0, $1+0}' \
  | sort -rn | cut -d" " -f2 | paste -sd,)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reclaim fragmentation
echo "==> Using GPUs $CUDA_VISIBLE_DEVICES (weights sharded across the roomy ones)"

echo "==> Host: $(hostname)"
echo "==> HF_HOME: $HF_HOME"
nvidia-smi || true

source "${DPR_VENV:-venv}/bin/activate"

# training-only deps (idempotent; skips if already present)
python -c "import peft, datasets, accelerate" 2>/dev/null || \
    pip install -q peft datasets accelerate

echo "==> Model: $MODEL | epochs: $EPOCHS"
echo "==> Validating data + loss-masking first:"
python -m training.sft_train --check-data --model "$MODEL"

echo "==> Training (adapter -> $OUT/adapter):"
python -m training.sft_train --model "$MODEL" --epochs "$EPOCHS" --out "$OUT"

echo "==> Done. LoRA adapter in $OUT/adapter/"
