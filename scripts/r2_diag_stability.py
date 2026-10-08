"""R2 段階4: 補助点が複数ある問題で、前の図にあった点が次の図でも同じ位置にあるかを調べる。

usage: python r2_diag_stability.py <in.jsonl> [--limit N]
"""
import json
import sys

from newclid.api import GeometricSolverBuilder
from vlma.figures import construction_texts, find_placement_seed, problem_text


def coords(text, seed):
    proof = GeometricSolverBuilder(seed=seed).load_problem_from_txt(text).build().proof
    return {n: (p.num.x, p.num.y) for n, p in proof.symbols_graph.name2node.items() if hasattr(p.num, "x")}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 200
    n = moved = 0
    for rec in rows[:limit]:
        try:
            cons = construction_texts(rec)
        except ValueError:
            continue
        if len(cons) < 2:
            continue
        seed = find_placement_seed(rec)
        if seed is None:
            continue
        n += 1
        states = [coords(problem_text(rec, cons[:k]), seed) for k in range(len(cons) + 1)]
        for a, b in zip(states, states[1:]):
            if any(abs(a[k][0] - b[k][0]) > 1e-9 or abs(a[k][1] - b[k][1]) > 1e-9 for k in a):
                moved += 1
                break
    print(json.dumps({"problems_with_2plus_constructs": n, "problems_where_a_point_moved": moved}))


if __name__ == "__main__":
    main()
