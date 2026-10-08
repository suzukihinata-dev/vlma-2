"""R2: 保存した証明の各手が、「その手が書いた根拠の事実だけを前提にして DDAR で導ける」を満たすか調べる。

usage: python r2_diag_cited.py <problems.jsonl> [--limit N]
各手について、根拠の番号を事実に直し、その事実だけを前提にして C++ 版 DDAR を max_level = 1, 2, 3, 10 で実行し、
結論が導けた割合を数える。段数を小さくするほど、長い推論を1手に押し込む近道を防げる。
"""
import json
import sys
import time
from collections import Counter

from newclid.DDAR.build import DDAR
from newclid.configs import load_solver_config
from newclid.discovery.extraction import parsing

from vlma.actions import Action, parse_action
from vlma.steps import expand

LEVELS = (1, 2, 3, 10)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 30
    cfg = load_solver_config(None)
    tally, no_cite, unknown_id, n_steps, t_run = Counter(), 0, 0, 0, 0.0
    fails = []
    n = 0
    for line in open(sys.argv[1]):
        c = json.loads(line)
        reg = {}
        body = parsing.strip_tags(c["problem"]).split("?", 1)[0]
        for pred, args, fid in parsing.parse_fact_segments(body):
            reg[fid] = (pred, list(args))
        fig_idx = 0
        for s in expand(c):
            a = parse_action(s["output"])
            if a.kind == "construct":
                for pred, args, fid in parsing.parse_fact_segments(a.body):
                    reg[fid] = (pred, list(args))
                fig_idx += 1
                continue
            pts = [(n_, xy[0], xy[1]) for n_, xy in c["figure_points"][fig_idx].items()]
            n_steps += 1
            cited = [reg[p] for p in a.premises if p in reg]
            if len(cited) < len(a.premises):
                unknown_id += 1
            if not cited:
                no_cite += 1
            goal = [(a.pred, list(a.args))]
            for L in LEVELS:
                t0 = time.time()
                solved, _ = DDAR.run_ddar("x", pts, cited, goal, L, cfg)
                t_run += time.time() - t0
                tally[L] += bool(solved)
                if L == 10 and not solved and len(fails) < 4:
                    fails.append((a.format()[:100], len(cited)))
            reg[a.fact_id] = (a.pred, list(a.args))
        n += 1
        if n >= limit:
            break
    print(json.dumps({"problems": n, "apply_steps": n_steps, "steps_citing_nothing": no_cite,
                      "steps_with_unknown_cited_id": unknown_id,
                      **{f"derivable_at_level_{L}": f"{tally[L]} ({tally[L] / n_steps:.1%})" for L in LEVELS},
                      "ddar_calls": n_steps * len(LEVELS), "mean_ms_per_call": round(1000 * t_run / (n_steps * len(LEVELS)), 1)}))
    print("not derivable at level 10, examples:", fails)


if __name__ == "__main__":
    main()
