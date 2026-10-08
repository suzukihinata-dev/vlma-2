#!/usr/bin/env bash
# R2 probe: 補助作図を含む問題を少量生成し、所要時間を記録する。
# usage: r2_gen_probe.sh <out_name> <n_samples> [extra pipeline args...]
set -euo pipefail
out=/artifacts/genesisgeo/r2/$1; n=$2; shift 2
cd /opt/genesisgeo
mkdir -p "$out"
start=$(date +%s)
timeout "${GEN_TIMEOUT:-1800}" python src/newclid/generation/pipeline.py \
  --n_clauses "${N_CLAUSES:-10}" --n_samples "$n" --n_threads "${N_THREADS:-4}" --max_level 100 \
  --aux_only 2 --seed_cache --dir "$out" "$@" > "$out.log" 2>&1 || echo "exit=$?"
end=$(date +%s)
echo "elapsed_s=$((end-start))"
find "$out" -name '*.jsonl' -exec wc -l {} +
tail -5 "$out.log"
