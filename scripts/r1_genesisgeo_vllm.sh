#!/usr/bin/env bash
set -euo pipefail

cd /opt/genesisgeo
python -c 'from scripts.launch_vllm_server import ensure_startup_patch; ensure_startup_patch()'

exec python -m vllm.entrypoints.cli.main serve /models/genesisgeo-2b \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.75 \
  --generation-config vllm \
  --max-model-len 4096 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --max-num-seqs 8 \
  --enforce-eager
