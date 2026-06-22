#!/bin/bash
set -ex
cd /home/qixinx/miles
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export PYTHONPATH=/home/qixinx/miles
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# Qwen3-4B-Instruct-2507 是标准 dense 模型(无 GDN/MTP),用其模型参数直接转 torch_dist。
source scripts/models/qwen3-4B-Instruct-2507.sh
torchrun --nproc-per-node 1 --master-port 23519 \
  tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
  --hf-checkpoint /data/user_data/qixinx/Qwen3-4B-Instruct-2507 \
  --save /data/user_data/qixinx/Qwen3-4B-Instruct-2507_torch_dist
