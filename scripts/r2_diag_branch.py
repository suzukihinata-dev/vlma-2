"""R2 段階4: 補助点の置き場所が、乱数 seed で変わるかを調べる。

usage: python r2_diag_branch.py <rejected.jsonl> [--n N]
"""
import json
import sys

from newclid.api import GeometricSolverBuilder
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.discovery.extraction import parsing
from vlma.steps import extract_actions


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    rows = [r for r in rows if r["reason"] == "not_hold"]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 2
    for r in rows[:n]:
        rec = r["record"]
        a = [x for x in extract_actions(rec) if x.kind == "construct"][0]
        cons = try_full_aux_dsl_to_constructions(a.to_genesis())
        prem, goal = rec["fl_problem"].split("?", 1)
        text = f"{prem.strip().rstrip(';')}; {cons} ? {goal.strip()}"
        facts = parsing.parse_fact_segments(parsing.extract_tag_content(rec["llm_output_renamed"], "numerical_check"))
        print("construction:", cons)
        for seed in range(8):
            solver = GeometricSolverBuilder(seed=seed).load_problem_from_txt(text).build()
            pt = solver.proof.symbols_graph.name2node[a.point].num
            ok = []
            for p, args, _ in facts:
                from newclid.statement import Statement
                st = Statement.from_tokens((p,) + tuple(args), solver.proof.dep_graph)
                ok.append(bool(st.check_numerical()) if st else None)
            print(f"  seed {seed}: {a.point} = ({pt.x:.4f}, {pt.y:.4f}) facts_ok={ok}")


if __name__ == "__main__":
    main()
