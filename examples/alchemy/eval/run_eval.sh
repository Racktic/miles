#!/usr/bin/env bash
# Zero-shot, no-summary Alchemy eval for frontier models (TMLR-aligned).
# API keys are read from $REPO/.env by llm_client.load_dotenv_keys() — NOT passed via --env, so they
# never appear in the process list. Trajectory/results land under examples/alchemy/logs/eval-*/ (in /home).
#
#   examples/alchemy/eval/run_eval.sh --provider anthropic --model claude-opus-4-8 --n-episodes 10
#   examples/alchemy/eval/run_eval.sh --n-episodes 1            # quick 1-episode smoke
set -euo pipefail
SIF=/data/user_data/qixinx/images/miles_dev-202606081341.sif
REPO=/home/qixinx/miles
USERBASE=/data/user_data/qixinx/arc_userbase
cd "$REPO"
exec apptainer exec \
  --env PYTHONUSERBASE="$USERBASE" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  "$SIF" python examples/alchemy/eval/eval_alchemy.py "$@"
