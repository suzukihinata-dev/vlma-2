"""R2: 保存した証明の各手が、「DDAR の記録した根拠が、すでに述べた事実の中にある」を満たすか調べる。

usage: python r2_diag_deps.py <problems.jsonl> [--limit N]
根拠なし（前提・作図・数値的な事実）の事実を最初の「述べた事実」とし、手を順に足しながら、
各手の結論について DDAR の記録した根拠が全て述べた事実に入っているかを数える。
"""
import json
import sys
from collections import Counter

from newclid.DDAR.build import DDAR
from newclid.api import CSolver, GeometricSolverBuilder
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.statement import Statement

from vlma.actions import Action, parse_action


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 50
    tally, reasons, n = Counter(), Counter(), 0
    examples = []
    for line in open(sys.argv[1]):
        c = json.loads(line)
        acts = [parse_action(a) for a in c["actions"]]
        cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if isinstance(a, Action) and a.kind == "construct"]
        prem, goal = c["fl_problem"].split("?", 1)
        text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
        solver = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build()
        cs = CSolver(solver=solver)
        _, dep = DDAR.run_ddar("x", cs.points, cs.premises, cs.goals, 500, cs.config)
        dg = solver.proof.dep_graph

        def key(tokens):
            try:
                return Statement.from_tokens(tuple(tokens), dg)
            except Exception:  # noqa: BLE001
                return None

        entries, established = {}, set()
        for stmt, deps, reason in dep:
            k = key(stmt)
            if k is None:
                continue
            entries[k] = (deps, reason)
            reasons[reason if not deps else "(derived)"] += 1
            if not deps:
                established.add(k)
        for a in acts:
            if not (isinstance(a, Action) and a.kind == "apply"):
                continue
            k = key(a.conclusion_tokens)
            if k is None or k not in entries:
                tally["conclusion_not_in_closure"] += 1
                continue
            deps = [key(d) for d in entries[k][0]]
            missing = [d for d in deps if d is None or d not in established]
            tally["deps_established" if not missing else "deps_missing"] += 1
            if missing and len(examples) < 3:
                examples.append((a.format()[:90], len(missing), len(deps)))
            established.add(k)
        n += 1
        if n >= limit:
            break
    print(json.dumps({"problems": n, **tally}, ensure_ascii=False))
    print("reasons of closure entries (top):", dict(reasons.most_common(6)))
    print("missing examples:", examples)


if __name__ == "__main__":
    main()
