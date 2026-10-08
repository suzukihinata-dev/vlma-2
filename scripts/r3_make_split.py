"""R3: 訓練/検証/テストの分割を作る。作図の並びが同じ問題（family）は、必ず同じ側に入れる。

usage: python r3_make_split.py --dir DIR [--val 1000] [--test 1000] [--per-family-cap 20] [--max-family-size 100] [--seed 0]
family = 前提の作図の並び（座標を除く）+ 補助作図。同じ family の問題は、同じ図の描き方で、目標だけが違う。
検証・テストの family は、大きすぎる family（--max-family-size 超）を避け、1つの family から --per-family-cap 問までにする。
選んだ family の残りの問題は、訓練にも使わない（excluded）。出力: DIR/split.json（問題 id → train / val / test / excluded）と、件数の内訳。
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

COORD = re.compile(r"@[-0-9.eE]+_[-0-9.eE]+")


def arg(name, default, cast=int):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def family(c):
    prem = COORD.sub("", c["fl_problem"].split("?", 1)[0])
    aux = [a for a in c["actions"] if a.startswith("CONSTRUCT")]
    return hashlib.sha1((prem + "|" + "|".join(aux)).encode()).hexdigest()[:12]


def main():
    d = Path(sys.argv[sys.argv.index("--dir") + 1])
    n_val, n_test = arg("--val", 1000), arg("--test", 1000)
    cap, max_size, seed = arg("--per-family-cap", 20), arg("--max-family-size", 100), arg("--seed", 0)
    fams = collections.defaultdict(list)
    for line in open(d / "problems.jsonl"):
        c = json.loads(line)
        fams[family(c)].append(c["problem_id"])
    order = sorted(fams, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
    split, counts = {}, collections.Counter()
    chosen = {"val": [], "test": []}
    for name, target in (("val", n_val), ("test", n_test)):
        got = 0
        for k in order:
            if got >= target:
                break
            if k in split or len(fams[k]) > max_size:
                continue
            members = sorted(fams[k], key=lambda p: hashlib.sha1(f"{seed}:{p}".encode()).hexdigest())
            for p in members[:cap]:
                split[p] = name
            for p in members[cap:]:
                split[p] = "excluded"
            split[k] = name  # family 自体の印（問題 id と衝突しない: 長さが同じなので、下で取り除く）
            got += min(len(members), cap)
            chosen[name].append(k)
    for k, members in fams.items():
        if k in chosen["val"] or k in chosen["test"]:
            continue
        for p in members:
            split[p] = "train"
    for k in chosen["val"] + chosen["test"]:
        split.pop(k, None)
    for p, s in split.items():
        counts[s] += 1
    json.dump(split, open(d / "split.json", "w"))
    out = {"problems": len(split), "counts": dict(counts), "families_total": len(fams),
           "families_val": len(chosen["val"]), "families_test": len(chosen["test"]),
           "args": {"val": n_val, "test": n_test, "per_family_cap": cap, "max_family_size": max_size, "seed": seed}}
    json.dump(out, open(d / "split_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
