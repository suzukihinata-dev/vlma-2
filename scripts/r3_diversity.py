"""R3: データの多様性を調べる。図（座標つき）と描き方（座標を除く）の数、1つの図から何問できているか。

usage: python r3_diversity.py <problems.jsonl>
"""
import collections
import hashlib
import json
import re
import sys

COORD = re.compile(r"@[-0-9.eE]+_[-0-9.eE]+")
sk_of_fig = collections.defaultdict(set)
fig_count, sk_count = collections.Counter(), collections.Counter()
clauses = collections.Counter()
ctypes = collections.Counter()
n = 0
for line in open(sys.argv[1]):
    c = json.loads(line)
    prem = c["fl_problem"].split("?", 1)[0]
    fig = hashlib.sha1(prem.encode()).hexdigest()[:12]
    sk = hashlib.sha1((COORD.sub("", prem) + "|" + "|".join(a for a in c["actions"] if a.startswith("CONSTRUCT"))).encode()).hexdigest()[:12]
    fig_count[fig] += 1
    sk_count[sk] += 1
    sk_of_fig[sk].add(fig)
    clauses[prem.count(";") + 1] += 1
    for cl in prem.split(";"):
        m = re.search(r"=\s*([a-z_0-9]+)", cl)
        if m:
            ctypes[m.group(1)] += 1
    n += 1
figs, sks = len(fig_count), len(sk_count)
sizes = sorted(fig_count.values())
q = lambda a, p: a[int(p * (len(a) - 1))]
print(json.dumps({"problems": n, "distinct_figures_with_coordinates": figs, "distinct_skeletons": sks,
                  "problems_per_figure": {"mean": round(n / figs, 1), "median": q(sizes, .5), "p90": q(sizes, .9), "max": sizes[-1]},
                  "figures_per_skeleton_mean": round(figs / sks, 1),
                  "single_goal_figures": sum(1 for s in sizes if s == 1)}))
print("clauses per problem (premises):", sorted(clauses.items()))
print("distinct construction names:", len(ctypes), "| top:", ctypes.most_common(8))
