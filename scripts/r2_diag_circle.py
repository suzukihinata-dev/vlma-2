"""R2: 結論が circle の手を、根拠の事実だけから C++ 版 DDAR で導けるか調べる。

usage: python r2_diag_circle.py <problems.jsonl> [--limit N] [--skip M]
"""
import json
import re
import sys

from newclid.DDAR.build import DDAR
from newclid.configs import load_solver_config

from vlma.actions import Action, parse_action
from vlma.steps import expand

PRED_COUNT = {}


def facts(text):
    return [(m[1], m[2].split(), m[3]) for m in re.finditer(r"([A-Za-z_]\w*)((?: (?:[a-z]\d*|\d[\w/.]*|pi[\w/.]*))+) \[(\d+)\]", text)]


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 400
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    cfg = load_solver_config(None)
    out = {}
    n = 0
    for i, line in enumerate(open(sys.argv[1])):
        if i < skip:
            continue
        n += 1
        if n > limit:
            break
        c = json.loads(line)
        reg = {fid: (p, a) for p, a, fid in facts(c["problem"].split("?")[0])}
        fig = 0
        for s in expand(c):
            a = parse_action(s["output"])
            if a.kind == "construct":
                reg.update({fid: (p, ar) for p, ar, fid in facts(a.body + " [999]" if False else a.body)})
                fig += 1
                continue
            pts = [(k, v[0], v[1]) for k, v in c["figure_points"][fig].items()]
            cited = [reg[p] for p in a.premises if p in reg]
            ok = DDAR.run_ddar("x", pts, cited, [(a.pred, list(a.args))], 1, cfg)[0] if len(cited) == len(a.premises) else None
            d = out.setdefault(a.pred, [0, 0, 0])
            d[0] += 1
            d[1] += bool(ok)
            d[2] += ok is None
            reg[a.fact_id] = (a.pred, list(a.args))
    bad = {k: v for k, v in out.items() if v[1] < v[0]}
    print("per predicate [steps, derivable_from_cited_at_level1, missing_cited_ids] (not all derivable only):", bad)
    print("totals:", sum(v[0] for v in out.values()), sum(v[1] for v in out.values()))


if __name__ == "__main__":
    main()
