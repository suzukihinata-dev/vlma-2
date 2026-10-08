"""R2: 保存した証明で rule_mismatch になる最初の手を探し、設定を変えたときに導けるかを調べる。

usage: python r2_diag_mismatch.py <problems.jsonl> --skip M [--limit N]
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
from vlma.verify import StepVerifier

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
for i, line in enumerate(open(sys.argv[1])):
    if i < skip:
        continue
    if i >= skip + limit:
        break
    c = json.loads(line)
    outs = [s["output"] for s in expand(c)]
    v = StepVerifier({"fl_problem": c["fl_problem"], "llm_input_renamed": c["problem"]}, seed=c["placement_seed"])
    bad = None
    for k, o in enumerate(outs):
        x = v.check(o)
        if not x.ok:
            bad = (k, o, x.reason)
            break
    if bad is None or bad[2] != "rule_mismatch":
        continue
    k, o, _ = bad
    acts = [parse_action(t) for t in outs]
    cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if a.kind == "construct"]
    prem, goal = c["fl_problem"].split("?", 1)
    text = "; ".join([prem.strip().rstrip(";"), *cons]) + f" ? {goal.strip()}"
    proof = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build().proof
    pts = [(n, p.num.x, p.num.y) for n, p in proof.symbols_graph.name2node.items() if hasattr(p.num, "x")]
    reg = {fid: (p, list(a)) for p, a, fid in parse_facts(c["problem"].split("?")[0])}
    for a in acts[:k]:
        if a.kind == "construct":
            reg.update({fid: (p, list(ar)) for p, ar, fid in parse_facts(a.body)})
        else:
            reg[a.fact_id] = (a.pred, list(a.args))
    a = acts[k]
    code = rules.rule_code(a.rule)
    cited = [reg[p] for p in a.premises]
    goal_ = [(a.pred, list(a.args))]
    print("problem", i, "step", k + 1, ":", o[:140])
    print("  cited:", cited)
    for name, cfg, lvl in [("rule only L1", cfg_with((code,)), 1), ("rule only L2", cfg_with((code,)), 2),
                           ("rule + AR L1", cfg_with((code,), True), 1), ("rule + AR L2", cfg_with((code,), True), 2),
                           ("all rules L1", BASE, 1)]:
        print("  ", name, bool(DDAR.run_ddar("x", pts, cited, goal_, lvl, cfg)[0]))
    break
