#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the target model checkpoint}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to the DFlash2 checkpoint}"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_TIMEOUT_KEEP_ALIVE=1800
exec python -m sglang.launch_server \
  --host 127.0.0.1 \
  --model-path "$MODEL_PATH" \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --tp-size 8 --ep-size 8 \
  --quantization modelopt_fp4 --moe-runner-backend humming \
  --mem-fraction-static 0.85 --swa-full-tokens-ratio 0.35 \
  --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
  --speculative-draft-attention-backend fa4 --mm-feature-transport cpu \
  --enable-hierarchical-cache --hicache-size 160 --chunked-prefill-size 16384 \
  --context-length 131072 --enable-cache-report --incremental-streaming-output \
  --random-seed 1234 --max-running-requests 32 --cuda-graph-max-bs-decode 32
