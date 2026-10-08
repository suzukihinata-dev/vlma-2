"""R3: 生成データの、独立な図（乱数の種）の数と、最初の作図の分布を調べる。

usage: python r3_seed_diversity.py <gen.jsonl> [<gen2.jsonl> ...]
"""
import collections
import json
import re
import sys

for path in sys.argv[1:]:
    seeds = collections.Counter()
    first = collections.Counter()
    first_by_seed = {}
    goals = collections.Counter()
    n = 0
    for line in open(path):
        r = json.loads(line)
        seeds[r["seed"]] += 1
        m = re.search(r"=\s*([a-z_0-9]+)", r["fl_problem"])
        if m:
            first[m.group(1)] += 1
            first_by_seed[r["seed"]] = m.group(1)
        g = r["fl_problem"].split("?")[-1].split()
        goals[g[0] if g else "?"] += 1
        n += 1
    per = sorted(seeds.values())
    by_seed = collections.Counter(first_by_seed.values())
    q = lambda a, p: a[int(p * (len(a) - 1))]
    top = lambda c, k: [(a, round(100 * b / sum(c.values()), 1)) for a, b in c.most_common(k)]
    print(path.split("/")[-3] + ":", json.dumps({"problems": n, "independent_seeds": len(seeds),
          "problems_per_seed": {"mean": round(n / len(seeds), 1), "median": q(per, .5), "p90": q(per, .9), "max": per[-1]}}))
    print("  first construction, by problems (%):", top(first, 5))
    print("  first construction, by seeds (%):   ", top(by_seed, 5), "| distinct:", len(by_seed))
    print("  goal predicate (%):", top(goals, 4))
