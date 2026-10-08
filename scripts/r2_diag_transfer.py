"""R2: Transfer の手が rule_mismatch で却下される問題を調べる。却下される最初の手の根拠の事実と、設定を変えたときの結果を出す。

usage: python r2_diag_transfer.py <problems.jsonl> --skip M [--limit N]
"""
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


def cfg_with(on=(), ar=False):
    cfg = dict(BASE)
    for k in list(cfg):
        if re.fullmatch(r"r\d+", k):
            cfg[k] = k in on
    cfg["using_ar"] = ar
    return cfg


skip = int(sys.argv[sys.argv.index("--skip") + 1])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 250
shown = 0
for i, line in enumerate(open(sys.argv[1])):
    if i < skip:
        continue
    if i >= skip + limit or shown >= 2:
        break
    c = json.loads(line)
    acts = [parse_action(s["output"]) for s in expand(c)]
    if not any(a.kind == "apply" and a.rule == "Transfer_between_equal_points" for a in acts):
        continue
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
        if a.rule == "Transfer_between_equal_points":
            res = {
                "all rules": bool(DDAR.run_ddar("x", pts, cited, goal_, 1, BASE)[0]),
                "no rules, no AR": bool(DDAR.run_ddar("x", pts, cited, goal_, 1, cfg_with())[0]),
                "AR only": bool(DDAR.run_ddar("x", pts, cited, goal_, 1, cfg_with(ar=True))[0]),
                "eqpoint rules r107-r109": bool(DDAR.run_ddar("x", pts, cited, goal_, 1, cfg_with(on=("r107", "r108", "r109")))[0]),
                "all rules, level 2": bool(DDAR.run_ddar("x", pts, cited, goal_, 2, BASE)[0]),
            }
            if not res["no rules, no AR"]:
                print("problem", i, "step:", a.format()[:120])
                print("  cited facts:", cited)
                print("  results:", res)
                shown += 1
                break
        reg[a.fact_id] = (a.pred, list(a.args))
