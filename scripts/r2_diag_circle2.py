"""R2: circle を結論とする手を、根拠だけから導けるか。目標として渡す方法と、導出の一覧から探す方法を比べる。

usage: python r2_diag_circle2.py <problems.jsonl> [--limit N] [--skip M]
"""
import json
import sys
from collections import Counter

from newclid.DDAR.build import DDAR
from newclid.api import GeometricSolverBuilder
from newclid.configs import load_solver_config
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.statement import Statement

from vlma.actions import Action, parse_action, parse_facts
from vlma.steps import expand


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 200
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    cfg = load_solver_config(None)
    t = Counter()
    shown = 0
    n = 0
    for i, line in enumerate(open(sys.argv[1])):
        if i < skip:
            continue
        n += 1
        if n > limit:
            break
        c = json.loads(line)
        steps = expand(c)
        if not any(parse_action(s["output"]).pred == "circle" for s in steps if s["output"].startswith("APPLY")):
            continue
        acts = [parse_action(s["output"]) for s in steps]
        cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if a.kind == "construct"]
        prem, goal = c["fl_problem"].split("?", 1)
        text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
        solver = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build()
        dg = solver.proof.dep_graph
        pts = [(k, p.num.x, p.num.y) for k, p in solver.proof.symbols_graph.name2node.items() if hasattr(p.num, "x")]
        reg = {fid: (p, list(a)) for p, a, fid in parse_facts(c["problem"].split("?")[0])}
        for a in acts:
            if a.kind == "construct":
                reg.update({fid: (p, list(ar)) for p, ar, fid in parse_facts(a.body)})
                continue
            if a.pred == "circle":
                cited = [reg[p] for p in a.premises]
                goal_based = DDAR.run_ddar("x", pts, cited, [(a.pred, list(a.args))], 1, cfg)[0]
                res = {}
                for L in (1, 2):
                    _, dep = DDAR.run_ddar("x", pts, cited, [], L, cfg)
                    want = Statement.from_tokens(a.conclusion_tokens, dg)
                    found = False
                    for stmt, deps, reason in dep:
                        try:
                            st = Statement.from_tokens(tuple(stmt), dg)
                        except Exception:  # noqa: BLE001
                            continue
                        if st is not None and want is not None and st.predicate.to_str(st) == want.predicate.to_str(want):
                            found = True
                            break
                    res[L] = found
                t["steps"] += 1
                t["goal_based"] += bool(goal_based)
                t["in_closure_level1"] += res[1]
                t["in_closure_level2"] += res[2]
                if shown < 3:
                    shown += 1
                    print(a.format(), "| cited:", cited, "| goal-based:", goal_based, "| closure:", res)
            reg[a.fact_id] = (a.pred, list(a.args))
    print(dict(t))


if __name__ == "__main__":
    main()
