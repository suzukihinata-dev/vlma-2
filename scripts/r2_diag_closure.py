"""R2: C++ 版 DDAR が出す導出の全体像（閉包）と、保存した証明の関係を調べる。

usage: python r2_diag_closure.py <problems.jsonl> [--limit N]
各問題で、保存した作図を足した問題に C++ 版 DDAR を実行し、導出された事実の数、目標に必要な事実の数、
1つの事実に複数の導出が記録されているか、保存した証明の手数との違いを出す。
"""
import json
import sys
from collections import Counter, defaultdict

from newclid.api import CSolver, GeometricSolverBuilder
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.configs import load_solver_config
from newclid.DDAR.build import DDAR

from vlma.actions import Action, parse_action


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 20
    rows = []
    for line in open(sys.argv[1]):
        rows.append(json.loads(line))
        if len(rows) >= limit:
            break
    out = []
    for c in rows:
        acts = [parse_action(a) for a in c["actions"]]
        cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if isinstance(a, Action) and a.kind == "construct"]
        prem, goal = c["fl_problem"].split("?", 1)
        text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
        solver = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build()
        cs = CSolver(solver=solver)
        solved, dep = DDAR.run_ddar("x", cs.points, cs.premises, cs.goals, 500, cs.config)
        stmts = defaultdict(list)
        for entry in dep:
            stmt, deps, reason = entry
            stmts[tuple(stmt) if isinstance(stmt, (list, tuple)) else stmt].append((deps, reason))
        multi = sum(len(v) > 1 for v in stmts.values())
        n_steps = sum(isinstance(a, Action) and a.kind == "apply" for a in acts)
        out.append({"id": c["problem_id"], "solved": solved, "entries": len(dep), "distinct_statements": len(stmts),
                    "statements_with_multiple_derivations": multi, "recorded_proof_steps": n_steps})
    print(json.dumps(out[:6], ensure_ascii=False))
    n = len(out)
    print(json.dumps({"problems": n,
                      "mean_entries": sum(o["entries"] for o in out) / n,
                      "mean_recorded_steps": sum(o["recorded_proof_steps"] for o in out) / n,
                      "problems_with_multiple_derivations": sum(o["statements_with_multiple_derivations"] > 0 for o in out)}))
    entry = dep[0] if dep else None
    print("sample entry:", repr(entry)[:300])


if __name__ == "__main__":
    main()
