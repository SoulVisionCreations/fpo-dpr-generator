#!/usr/bin/env bash
# Run the DPR endpoints bare-metal (no Docker).
#
# This script exists for one reason: the Dockerfile carried the environment that
# points the service at merged, local weights. Run uvicorn by hand without it and
# DPR_MODEL_ID still defaults to "google/gemma-4-12B-it", so the service quietly
# starts downloading 23 GB from Hugging Face and enables LoRA machinery that the
# merged weights do not need. That has filled a disk twice.
#
#   ./scripts/serve.sh                      # weights at /opt/dpr-weights
#   DPR_WEIGHTS=~/dpr-weights ./scripts/serve.sh
#   DPR_API_KEY=secret ./scripts/serve.sh   # require a key (default: open)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WEIGHTS="${DPR_WEIGHTS:-/opt/dpr-weights}"

# Fail here, loudly, rather than 20 minutes into an unintended download.
if ! compgen -G "$WEIGHTS/*.safetensors" >/dev/null; then
  echo "error: no .safetensors in $WEIGHTS" >&2
  echo "       These are the MERGED weights (adapter already folded in):" >&2
  echo "         mkdir -p $WEIGHTS && cd $WEIGHTS" >&2
  echo "         oras pull ghcr.io/soulvisioncreations/dpr-weights:v4" >&2
  echo "       Set DPR_WEIGHTS to point somewhere else." >&2
  exit 1
fi

export DPR_MODEL_ID="$WEIGHTS"
export DPR_MERGED_MODEL=1          # adapter is folded in; no LoRA to load
export HF_HUB_OFFLINE=1            # nothing should reach the Hub at runtime
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DPR_MAX_MODEL_LEN="${DPR_MAX_MODEL_LEN:-8192}"
export DPR_GPU_MEMORY_UTILIZATION="${DPR_GPU_MEMORY_UTILIZATION:-0.90}"
export DPR_DATA_DIR="${DPR_DATA_DIR:-$REPO/data}"

# financial_engine._assert_balance is the only check that means of finance equals
# project cost, and `python -O` strips it. serve/app.py refuses to start under it;
# this just makes sure nothing in the environment turns it on.
unset PYTHONOPTIMIZE

# Open by default, which is correct behind a gateway on a private network.
# Setting DPR_API_KEY is enough to turn the key check back on.
export DPR_REQUIRE_AUTH="${DPR_REQUIRE_AUTH:-${DPR_API_KEY:+1}}"
export DPR_REQUIRE_AUTH="${DPR_REQUIRE_AUTH:-0}"

PORT="${DPR_PORT:-9000}"
echo "weights : $WEIGHTS"
echo "port    : $PORT   (auth: $([ "$DPR_REQUIRE_AUTH" = 1 ] && echo on || echo off))"
echo "first start loads ~22 GB; /ready returns 200 in 3-5 minutes"

# --workers 1: N workers would mean N engines and instant OOM.
# keep-alive above the 240s request deadline so the connection outlives a slow DPR.
exec python -m uvicorn serve.app:app \
  --host 0.0.0.0 --port "$PORT" \
  --workers 1 --timeout-keep-alive 305 --no-access-log
