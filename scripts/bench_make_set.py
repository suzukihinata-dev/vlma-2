"""ベンチマークの問題の一覧を作る。テスト分割から、決まった乱数で N 問を選ぶ（モデルによらず固定）。

usage: python bench_make_set.py --dir DIR [--n 200] [--seed 0] [--split test] [--out bench_200.json]
"""
import json
import random
import sys
from pathlib import Path


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


d = Path(arg("--dir"))
n, seed, split_name = arg("--n", 200, int), arg("--seed", 0, int), arg("--split", "test")
out = d / arg("--out", f"bench_{n}.json")
split = json.load(open(d / "split.json"))
meta = json.load(open(d / "turn_index.json"))
pids = sorted(p for p in meta if split.get(p) == split_name)
random.Random(seed).shuffle(pids)
chosen = pids[:n]
json.dump(chosen, open(out, "w"))
turns = sorted(meta[p][1] for p in chosen)
q = lambda p: turns[int(p * (len(turns) - 1))]
print(json.dumps({"split": split_name, "available": len(pids), "selected": len(chosen), "reference_turns": {"min": turns[0], "median": q(.5), "p90": q(.9), "max": turns[-1]},
                  "out": str(out)}))
