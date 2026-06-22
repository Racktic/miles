#!/bin/bash
set -ex
cd /home/qixinx/miles
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export PYTHONPATH=/home/qixinx/miles
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# HF->torch_dist via --spec get_qwen3_5_spec (a pure text GPT decoder): the stock VL+MTP config.json is
# ignored, so the stock public Qwen3.5-4B converts straight to a flat 378-tensor text ckpt. No MTP/vision
# stripping needed. Train with the SAME flat path (default raw mode, NOT --megatron-to-hf-mode bridge).
source scripts/models/qwen3.5-4B.sh
torchrun --nproc-per-node 1 --master-port 23517 \
  tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint /data/user_data/qixinx/Qwen3.5-4B \
  --save /data/user_data/qixinx/Qwen3.5-4B_torch_dist
