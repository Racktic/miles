#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=01:00:00
set -uo pipefail
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO PIP_CERT
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
echo "=== $(date) node=$(hostname) gpu=$(nvidia-smi -L | head -1) ==="
apptainer exec --nv --bind /data,/home/qixinx "$SIF" \
  python3 /home/qixinx/think_write_test/run_compare.py /home/qixinx/think_write_test/scenarios_v3.json /home/qixinx/think_write_test/results_v3.json
echo "JOB_DONE"
