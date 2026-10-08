"""R2: Transfer の手が、根拠に eqpoint の事実を含むかを数える（DDAR は使わない）。

usage: python r2_diag_transfer_cites.py <problems.jsonl> [--limit N]
"""
import collections
import json
import sys

from vlma.actions import parse_action, parse_facts
from vlma.steps import expand

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 100000
n_tr = with_eq = no_eq = 0
other = collections.Counter()
examples = []
for i, line in enumerate(open(sys.argv[1])):
    if i >= limit:
        break
    c = json.loads(line)
    reg = {fid: p for p, a, fid in parse_facts(c["problem"].split("?")[0])}
    for s in expand(c):
        a = parse_action(s["output"])
        if a.kind == "construct":
            reg.update({fid: p for p, ar, fid in parse_facts(a.body)})
            continue
        if a.rule == "Transfer_between_equal_points":
            n_tr += 1
            preds = [reg.get(p) for p in a.premises]
            if "eqpoint" in preds:
                with_eq += 1
            else:
                no_eq += 1
                other[tuple(sorted(str(x) for x in preds))] += 1
                if len(examples) < 3:
                    examples.append(s["output"][:120])
        reg[a.fact_id] = a.pred
print(json.dumps({"transfer_steps": n_tr, "cite_an_eqpoint": with_eq, "no_eqpoint": no_eq}))
print("without eqpoint:", other.most_common(3), examples)
