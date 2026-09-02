#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# author_v3.sh - STEP 2 (+ optional STEP 3) of the gpt-5 v3 dataset build.
#
# gpt-5 authors a full Option-B DPR for every instance in data/finetune/instances/
# (resumable: skips ids already in teacher_out/, so re-run after any interruption).
# API/CPU only -- no GPU, does NOT touch SLURM or the stress test.
#
#   ./author_v3.sh                      # author with gpt-5, concurrency 8
#   ./author_v3.sh gpt-5 12             # model=gpt-5, concurrency=12
#   ./author_v3.sh gpt-5 8 --collect    # author, THEN build train/val.jsonl
#
# Background it (survives your shell closing) and watch the log:
#   nohup ./author_v3.sh gpt-5 8 --collect > /dev/null 2>&1 &
#   tail -f data/finetune/logs/author_v3.log
#   ./training/slurm/pipeline_status.sh
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1   # repo root

MODEL="${1:-gpt-5}"
CONC="${2:-8}"
COLLECT=0
for a in "$@"; do [ "$a" = "--collect" ] && COLLECT=1; done

FT=data/finetune
LOG=$FT/logs/author_v3.log
mkdir -p "$FT/logs" "$FT/teacher_out"

# Load the OpenAI key.
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY not set (put it in .env)." >&2; exit 1
fi

INST=$(ls $FT/instances/*.json 2>/dev/null | wc -l)
DONE=$(ls $FT/teacher_out/*.md 2>/dev/null | wc -l)
echo "==> author_v3: model=$MODEL concurrency=$CONC | instances=$INST already-authored=$DONE"
echo "==> logging to $LOG  (tail -f to watch; safe to re-run to resume)"

PYTHONUNBUFFERED=1 ./venv/bin/python -m training.teacher \
    --provider openai --model "$MODEL" --concurrency "$CONC" \
    2>&1 | tee "$LOG"

DONE=$(ls $FT/teacher_out/*.md 2>/dev/null | wc -l)
echo "==> authored now: $DONE / $INST"

if [ "$COLLECT" = "1" ]; then
    echo "==> STEP 3: assemble+verify -> train/val.jsonl"
    ./venv/bin/python -m training.teacher --collect --val-frac 0.15 2>&1 | tee -a "$LOG"
    wc -l $FT/train.jsonl $FT/val.jsonl 2>/dev/null || true
    echo "==> Next: sbatch training/slurm/launch_sft.sh google/gemma-4-12B-it 3 v3gpt"
else
    echo "==> Next (build dataset): ./venv/bin/python -m training.teacher --collect --val-frac 0.15"
fi
