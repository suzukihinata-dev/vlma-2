"""R2: 手の結論を、根拠だけから導くのに必要な DDAR の段数を、述語ごとに調べる（丸めていない座標を使う）。

usage: python r2_diag_levels.py <problems.jsonl> [--limit N] [--skip M] [--only-circle]
"""
import json
import sys
from collections import defaultdict

from newclid.DDAR.build import DDAR
from newclid.api import GeometricSolverBuilder
from newclid.configs import load_solver_config
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions

from vlma.actions import parse_action, parse_facts
from vlma.steps import expand

LEVELS = (1, 2, 3)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 100
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    cfg = load_solver_config(None)
    table = defaultdict(lambda: [0, 0, 0, 0])  # 述語 -> [手数, level1, level2, level3]
    n = 0
    for i, line in enumerate(open(sys.argv[1])):
        if i < skip:
            continue
        n += 1
        if n > limit:
            break
        c = json.loads(line)
        acts = [parse_action(s["output"]) for s in expand(c)]
        cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if a.kind == "construct"]
        prem, goal = c["fl_problem"].split("?", 1)
        text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
        proof = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build().proof
        pts = [(k, p.num.x, p.num.y) for k, p in proof.symbols_graph.name2node.items() if hasattr(p.num, "x")]
        reg = {fid: (p, list(a)) for p, a, fid in parse_facts(c["problem"].split("?")[0])}
        for a in acts:
            if a.kind == "construct":
                reg.update({fid: (p, list(ar)) for p, ar, fid in parse_facts(a.body)})
                continue
            cited = [reg[p] for p in a.premises]
            row = table[a.pred]
            row[0] += 1
            for j, L in enumerate(LEVELS, start=1):
                row[j] += bool(DDAR.run_ddar("x", pts, cited, [(a.pred, list(a.args))], L, cfg)[0])
            reg[a.fact_id] = (a.pred, list(a.args))
    print("述語: [手数, 1段で導ける, 2段で導ける, 3段で導ける]（全て導けるものは省く）")
    for k, v in sorted(table.items(), key=lambda kv: -kv[1][0]):
        if v[1] < v[0] or v[3] < v[0]:
            print(" ", k, v)
    print("totals:", [sum(v[j] for v in table.values()) for j in range(4)])


if __name__ == "__main__":
    main()
