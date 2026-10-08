"""R2: 規則名の確認の識別力。同じ手に、別の規則名を付けたときに、通ってしまう割合を調べる。

usage: python r2_diag_rule_alt.py <problems.jsonl> [--limit N] [--skip M]
各手で、結論の述語を結論にできる他の規則（番号が違うもの）を、その規則だけを有効にして1段で実行する。
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
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 30
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    candidates = collections.defaultdict(list)  # 述語 -> 候補の名前（番号ごとに1つ）
    for name in rules.ALL_NAMES:
        code = rules.rule_code(name)
        if code == "Transfer":
            continue
        for pred in ("eqangle", "eqratio", "cong", "para", "perp", "coll", "cyclic", "simtri", "simtrir", "contri", "contrir", "midp", "circle", "rconst"):
            if rules.is_valid(name, pred) and code not in [rules.rule_code(x) for x in candidates[pred]]:
                candidates[pred].append(name)
    steps = alt_steps = 0
    pairs = collections.Counter()
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
            steps += 1
            passing = []
            for alt in candidates[a.pred]:
                if rules.rule_code(alt) == rules.rule_code(a.rule):
                    continue
                code = rules.rule_code(alt)
                ok = DDAR.run_ddar("x", pts, cited, goal_, 1, only(code))[0]
                if not ok and code != "AR":  # 規則と AR を併用すれば導け、AR だけでは導けない場合も認める（verify と同じ）
                    with_ar = only(code)
                    with_ar["using_ar"] = True
                    ok = DDAR.run_ddar("x", pts, cited, goal_, 1, with_ar)[0] and not DDAR.run_ddar("x", pts, cited, goal_, 1, only("AR"))[0]
                if ok:
                    passing.append(alt)
            if passing:
                alt_steps += 1
                for alt in passing:
                    pairs[(a.rule, alt)] += 1
            reg[a.fact_id] = (a.pred, list(a.args))
    print(json.dumps({"steps": steps, "steps_where_another_rule_also_derives_it": alt_steps,
                      "share": round(alt_steps / steps, 3)}))
    print("most common (written rule -> another rule that also passes):")
    for (r, alt), k in pairs.most_common(8):
        print(f"  {r}  ->  {alt}   {k}")


if __name__ == "__main__":
    main()
