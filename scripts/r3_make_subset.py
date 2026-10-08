"""R3: 学習に使う問題の一覧（subset）を作る。独立な図（乱数の種）ごとに1問、訓練の分割から選ぶ。

usage: python r3_make_subset.py --dir DIR --raw gen1.jsonl [--raw gen2.jsonl ...] --n 1000 [--per-seed 1] [--seed 0] [--out subset_1000.json]
       [--base subset_1000.json]   既存の一覧を先頭に置き、足りない分を、図（種）を順に回って1問ずつ足す（方式 B の、問題数を増やした一覧）
problems.jsonl のコンパクトな形には、生成機の乱数の種が無いので、生成機の出力（--raw）から、問題 id → 種を引く。
同じ種から作られた問題は、同じ図の別の部分や別の目標なので、1つの種から選ぶのは --per-seed 問までにして、図の多様性を保つ。
"""
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    d = Path(arg("--dir"))
    raws = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--raw"]
    n, per_seed, seed = arg("--n", 1000, int), arg("--per-seed", 1, int), arg("--seed", 0, int)
    out = d / arg("--out", f"subset_{n}.json")
    seed_of, first_of = {}, {}
    for pth in raws:
        for line in open(pth):
            r = json.loads(line)
            pid = hashlib.sha1(r["fl_problem"].encode()).hexdigest()[:12]
            seed_of[pid] = (pth, r["seed"])
            m = re.search(r"=\s*([a-z_0-9]+)", r["fl_problem"])
            first_of[pid] = m.group(1) if m else "?"
    split = json.load(open(d / "split.json"))
    meta = json.load(open(d / "turn_index.json"))
    by_seed = collections.defaultdict(list)
    for pid in meta:
        if split.get(pid) == "train" and pid in seed_of:
            by_seed[seed_of[pid]].append(pid)
    rng = random.Random(seed)
    seeds = sorted(by_seed)
    rng.shuffle(seeds)
    chosen = []
    base = arg("--base")
    if base:  # 先頭は既存の一覧。残りは、図を順に回って、まだ選んでいない問題を1問ずつ足す（どの図も同じ数に近づく）
        chosen = [p for p in json.load(open(d / base)) if p in meta]
        taken = set(chosen)
        queues = {}
        for s in seeds:
            members = sorted(by_seed[s])
            rng.shuffle(members)
            queues[s] = [m for m in members if m not in taken]
        while len(chosen) < n and any(queues.values()):
            for s in seeds:
                if queues[s] and len(chosen) < n:
                    chosen.append(queues[s].pop(0))
        if len(chosen) < n:
            print(f"warning: only {len(chosen)} problems available", file=sys.stderr)
    else:
        for s in seeds:
            members = sorted(by_seed[s])
            rng.shuffle(members)
            chosen.extend(members[:per_seed])
            if len(chosen) >= n:
                break
    chosen = chosen[:n]
    json.dump(chosen, open(out, "w"))
    turns = [meta[p][1] for p in chosen]
    firsts = collections.Counter(first_of[p] for p in chosen)
    summary = {"selected": len(chosen), "independent_seeds_available_in_train": len(by_seed), "per_seed": per_seed,
               "turns_per_problem": {"mean": round(sum(turns) / len(turns), 1), "min": min(turns), "max": max(turns)},
               "samples_per_epoch": sum(turns), "first_construction": [(k, round(100 * v / len(chosen), 1)) for k, v in firsts.most_common(8)],
               "distinct_first_constructions": len(firsts), "problems_per_seed_max": max(collections.Counter(seed_of[p] for p in chosen).values()),
               "out": str(out)}
    json.dump(summary, open(str(out).replace(".json", "_summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
