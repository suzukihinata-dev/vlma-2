#!/bin/bash
# ベンチマークの環境サーバーを K 個（ポート 8100〜）起動する。genesisgeo のコンテナで動かす（DDAR は CPU。サーバー1つにつき1コア）。
# usage: bench_env.sh <name> [K=6] [data-dir]   図は /artifacts/bench/<name>/figs-live に描く。data-dir の既定は、合成データのテスト分割
#   IMO-AG-30 など: bench_env.sh imo30 6 /artifacts/bench/imo30（bench_make_imo.py で作った問題）
NAME=${1:?name}; K=${2:-6}; DATA=${3:-/artifacts/genesisgeo/r2/build-100k}
mkdir -p /artifacts/bench/$NAME
for i in $(seq 0 $((K-1))); do
  python -m vlma.envserver --port $((8100+i)) --data-dir $DATA --out-dir /artifacts/bench/$NAME/figs-live >> /artifacts/bench/$NAME/env-$i.log 2>&1 &
done
echo "env servers: ports 8100-$((8100+K-1))"
wait
