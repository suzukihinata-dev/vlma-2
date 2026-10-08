"""R2: 保存した証明の (規則の番号, 結論の述語) の組と回数を数える。

usage: python r2_rule_pairs.py <problems.jsonl>
"""
import collections
import json
import sys

from vlma.actions import Action, parse_action

pairs = collections.Counter()
for line in open(sys.argv[1]):
    c = json.loads(line)
    for a in c["actions"]:
        p = parse_action(a)
        if isinstance(p, Action) and p.kind == "apply":
            pairs[(p.rule, p.pred)] += 1
total = sum(pairs.values())
print("distinct (rule, conclusion predicate) pairs:", len(pairs), "steps:", total)
for (r, pr), n in sorted(pairs.items(), key=lambda kv: (kv[0][0], -kv[1])):
    print(f"{r:9} {pr:9} {n:9}")
