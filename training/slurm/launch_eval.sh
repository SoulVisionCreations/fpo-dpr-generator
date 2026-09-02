#!/bin/bash
#SBATCH --job-name=dpr-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ---------------------------------------------------------------------------
# Evaluate a fine-tuned (or base) DPR model on the held-out val instances:
# generate -> render.assemble (inject exact tables + verify numbers) -> score §2.
#
#   sbatch launch_eval.sh google/gemma-4-12B-it v3gpt 3   # TAG -> models/<slug>_v3gpt/adapter
#   sbatch launch_eval.sh google/gemma-4-12B-it none 3    # base model, no adapter
#   sbatch launch_eval.sh <model> models/<slug>/adapter 3 # explicit adapter path also works
#
# Arg 1 = model id, Arg 2 = TAG (matches launch_sft.sh) | adapter path | "none"/"base",
# Arg 3 = #instances (default 3). Output: output/ft_eval/*.md + summary__*.json
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_ARG="${2:-none}"
LIMIT="${3:-3}"

cd "$(dirname "$0")/../.." || exit 1          # repo root
mkdir -p logs output/ft_eval

# Resolve arg 2 into an adapter path. Mirror launch_sft.sh's dir naming so the SAME
# tag works for both: SLUG=model with '/' and ':' -> '_', dir = models/<SLUG>_<TAG>.
SLUG=$(echo "$MODEL" | tr '/:' '__')
if [ "$ADAPTER_ARG" = "none" ] || [ "$ADAPTER_ARG" = "base" ]; then
    ADAPTER="none"
elif [ -d "$ADAPTER_ARG" ]; then
    ADAPTER="$ADAPTER_ARG"                            # explicit path given
else
    ADAPTER="models/${SLUG}_${ADAPTER_ARG}/adapter"   # treat arg2 as a tag
fi
if [ "$ADAPTER" != "none" ] && [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter dir not found: $ADAPTER" >&2
    echo "  (arg2 '$ADAPTER_ARG' resolved here; pass a valid tag, an adapter path, or 'none' for base)" >&2
    echo "  available:" >&2; ls -d models/*/adapter 2>/dev/null | sed 's/^/    /' >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# GPU selection: pin explicitly with EVAL_GPU (e.g. EVAL_GPU=1 sbatch ... to run two
# eval jobs in parallel on DIFFERENT GPUs). Otherwise auto-pick roomiest-first
# (bounded by DPR_MAX_GPU_INDEX, default: no bound).
if [ -n "${EVAL_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
else
    export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F',' -v max="${DPR_MAX_GPU_INDEX:-99}" '$1+0<=max {print $2+0, $1+0}' \
  | sort -rn | cut -d" " -f2 | paste -sd,)
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> Host: $(hostname)"
echo "==> HF_HOME: $HF_HOME"
echo "==> GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader || true

source "${DPR_VENV:-venv}/bin/activate"
# peft for adapters; pillow+timm are needed to instantiate multimodal models
# (gemma-3n / gemma-3-4b have a timm vision tower) even when we only feed text.
python -c "import peft, PIL, timm" 2>/dev/null || pip install -q peft pillow timm

echo "==> Model: $MODEL | adapter: $ADAPTER | instances: $LIMIT"
if [ "$ADAPTER" = "none" ]; then
    python -m training.eval_ft --model "$MODEL" --limit "$LIMIT"
else
    python -m training.eval_ft --model "$MODEL" --adapter "$ADAPTER" --limit "$LIMIT"
fi

echo "==> Done. Results in output/ft_eval/"