#!/usr/bin/env bash
# Standalone sglang OpenAI-compatible server for Qwen3.5-4B, for the Alchemy eval (inference only).
# The 4B model fits on ONE 80GB card, so to use multiple cards we DATA-parallel (DP replicas, tp=1
# each) — NOT tensor-parallel (TP would add comm overhead and slow a model this small). DP=4 gives ~4x
# throughput for concurrent eval requests (pair with a higher --workers on the eval side).
# Exposes /v1/chat/completions on $PORT. Point the eval at it:
#   run_eval.sh --provider openai --base-url http://127.0.0.1:30000/v1 --model qwen3.5-4b ...
#
#   DP=4 examples/alchemy/eval/serve_qwen.sh          # 4 replicas across GPUs 0-3 (default)
#   DP=1 GPUS=0 examples/alchemy/eval/serve_qwen.sh   # single GPU
set -euo pipefail
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
CKPT=/data/user_data/qixinx/Qwen3.5-4B   # re-downloaded 2026-06-15 (old qwen3.5-4B-nomtp symlinks died when /data/hf_cache blobs were wiped)
PORT=${PORT:-30000}
DP=${DP:-4}                        # data-parallel replicas
TP=${TP:-1}                        # tensor-parallel per replica (keep 1 for a 4B model)
GPUS=${GPUS:-0,1,2,3}              # cards to expose (need DP*TP of them)
echo "[serve_qwen] launching sglang for $CKPT on GPUs $GPUS port $PORT (dp=$DP tp=$TP)" >&2
exec apptainer exec --nv \
  --env CUDA_VISIBLE_DEVICES="$GPUS" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python -m sglang.launch_server \
    --model-path "$CKPT" \
    --served-model-name qwen3.5-4b \
    --tp "$TP" --dp-size "$DP" \
    --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static 0.5 \
    --trust-remote-code
