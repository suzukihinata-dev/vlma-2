"""R2 段階4: 検証で却下された問題を調べる。却下された手の規則名と、DDAR の実行状況を出す。

usage: python r2_diag_reject.py <in.jsonl> [--limit N] [--engine c|python]
"""
import collections
import json
import sys
import time

from newclid.api import GeometricSolverBuilder
from vlma.actions import Action, parse_action
from vlma.steps import to_steps
from vlma.verify import StepVerifier


ENGINE = sys.argv[sys.argv.index("--engine") + 1] if "--engine" in sys.argv else "c"


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rules_bad, rules_all = collections.Counter(), collections.Counter()
    n_bad = 0
    for i, rec in enumerate(rows[:limit]):
        steps = to_steps(rec)
        v = StepVerifier(rec, engine=ENGINE)
        bad = []
        for k, s in enumerate(steps):
            verdict = v.check(s["output"])
            a = parse_action(s["output"])
            if isinstance(a, Action) and a.kind == "apply":
                rules_all[a.rule] += 1
                if not verdict.ok:
                    bad.append((k + 1, a.rule, s["output"], verdict.reason))
                    rules_bad[a.rule] += 1
        if bad:
            n_bad += 1
            t0 = time.time()
            premises, goal = rec["fl_problem"].split("?", 1)
            cons = "; ".join(v._aux)
            text = f"{premises.strip().rstrip(';')}; {cons} ? {goal.strip()}"
            solver = GeometricSolverBuilder(seed=123).load_problem_from_txt(text).build()
            solved = solver.run(timeout=600)
            print(json.dumps({"i": i, "n_turns": len(steps), "n_bad_steps": len(bad), "first_bad": bad[0],
                              "goal_solved_by_ddar": solved, "run_s": round(time.time() - t0, 1),
                              "run_infos": {k: v for k, v in solver.run_infos.items() if k in ("runtime", "steps", "success")}},
                             ensure_ascii=False), flush=True)
    print("problems with bad steps:", n_bad, "of", len(rows[:limit]))
    print("bad by rule:", dict(rules_bad))
    print("rule frequency among all apply steps (top):", dict(rules_all.most_common(8)))


if __name__ == "__main__":
    main()
