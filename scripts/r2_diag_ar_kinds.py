"""R2: C++ 版 DDAR が付ける AR の種類（AR with Dist / Distlog / Slope）と、結論の述語の対応を数える。

usage: python r2_diag_ar_kinds.py <problems.jsonl> [--limit N] [--with-pred P]
--with-pred: その述語を結論に持つ AR の手がある問題だけを対象にする
"""
import collections
import json
import sys

from newclid.DDAR.build import DDAR
from newclid.api import CSolver, GeometricSolverBuilder
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions

from vlma.actions import Action, parse_action

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 40
tab = collections.Counter()
with_pred = sys.argv[sys.argv.index("--with-pred") + 1] if "--with-pred" in sys.argv else None
n_done = 0
for i, line in enumerate(open(sys.argv[1])):
    if n_done >= limit:
        break
    c = json.loads(line)
    if with_pred and not any(a.startswith("APPLY AR : " + with_pred + " ") for a in c["actions"]):
        continue
    n_done += 1
    acts = [parse_action(a) for a in c["actions"]]
    cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if isinstance(a, Action) and a.kind == "construct"]
    prem, goal = c["fl_problem"].split("?", 1)
    text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
    solver = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build()
    cs = CSolver(solver=solver)
    _, dep = DDAR.run_ddar("x", cs.points, cs.premises, cs.goals, 500, cs.config)
    for stmt, deps, reason in dep:
        if reason.startswith("AR") or reason == "Transfer":
            tab[(reason, stmt[0])] += 1
by_reason = collections.defaultdict(dict)
for (r, p), n in tab.items():
    by_reason[r][p] = n
for r, d in by_reason.items():
    print(r, dict(sorted(d.items(), key=lambda kv: -kv[1])))
