"""R2: 規則を番号から名前に変えたときの、手の出力と、最後のターンの過去の操作列の文字数の増え方を測る。

usage: python r2_name_cost.py <problems.jsonl> [--limit N]
"""
import json
import statistics
import sys

from vlma.actions import Action, parse_action
from vlma.steps import expand

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 2000
step_code, step_name, hist_code, hist_name = [], [], [], []
for n, line in enumerate(open(sys.argv[1])):
    if n >= limit:
        break
    c = json.loads(line)
    samples = expand(c)
    outs = [s["output"] for s in samples]
    codes = []
    for s in samples:
        a = parse_action(s["output"])
        codes.append(s["output"] if a.kind == "construct" else
                     "APPLY " + a.to_genesis().split("]")[1].split()[0] + s["output"].split(" ", 2)[2].replace(a.rule, "", 1)
                     if False else s["output"].replace(a.rule, __import__("vlma.rules", fromlist=["rule_code"]).rule_code(a.rule), 1))
    step_name += [len(o) for o in outs]
    step_code += [len(o) for o in codes]
    hist_name.append(sum(len(o) + 1 for o in outs[:-1]))
    hist_code.append(sum(len(o) + 1 for o in codes[:-1]))
q = lambda a, p: sorted(a)[int(p * (len(a) - 1))]
print(json.dumps({
    "problems": len(hist_name),
    "chars_per_step_code": round(statistics.mean(step_code), 1), "chars_per_step_name": round(statistics.mean(step_name), 1),
    "final_turn_history_chars_median": [q(hist_code, .5), q(hist_name, .5)],
    "final_turn_history_chars_p90": [q(hist_code, .9), q(hist_name, .9)],
    "final_turn_history_chars_max": [max(hist_code), max(hist_name)],
}, ensure_ascii=False))
