"""R2: 手に書いた規則「だけ」で、書いた根拠から結論が導けるかを、保存した証明で調べる。

usage: python r2_diag_rule_only.py <problems.jsonl> [--limit N] [--skip M]
C++ 版 DDAR の設定で、その規則（AR なら代数的推論だけ）以外をすべて無効にして、1段・2段で実行する。
"""
import collections
import json
import re
import sys

from newclid.DDAR.build import DDAR
from newclid.api import GeometricSolverBuilder
from newclid.configs import load_solver_config
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions

from vlma import rules
from vlma.actions import parse_action, parse_facts
from vlma.steps import expand

BASE = load_solver_config(None)


def only(code):
    cfg = dict(BASE)
    for k in list(cfg):
        if re.fullmatch(r"r\d+", k):
            cfg[k] = False
    cfg["using_ar"] = code == "AR"
    if re.fullmatch(r"r\d+", code):
        cfg[code] = True
    return cfg


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 50
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    tab = collections.defaultdict(lambda: [0, 0, 0, 0])  # 名前 -> [手数, 全規則で1段, その規則だけで1段, その規則だけで2段]
    fails = []
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
            goal_ = [(a.pred, list(a.args))]
            code = rules.rule_code(a.rule)
            row = tab[a.rule]
            row[0] += 1
            row[1] += bool(DDAR.run_ddar("x", pts, cited, goal_, 1, BASE)[0])
            cfg = only(code)
            ok1 = bool(DDAR.run_ddar("x", pts, cited, goal_, 1, cfg)[0])
            ok2 = ok1 or bool(DDAR.run_ddar("x", pts, cited, goal_, 2, cfg)[0])
            row[2] += ok1
            row[3] += ok2
            if not ok2 and len(fails) < 6:
                fails.append((a.rule, a.format()[:110]))
            reg[a.fact_id] = (a.pred, list(a.args))
    print("名前: [手数, 全規則で1段, その規則だけで1段, その規則だけで2段]（全て通るものは省く）")
    for k, v in sorted(tab.items(), key=lambda kv: -kv[1][0]):
        if v[2] < v[0] or v[3] < v[0]:
            print(" ", k, v)
    print("totals:", [sum(v[j] for v in tab.values()) for j in range(4)])
    print("failures:", fails)


if __name__ == "__main__":
    main()
