"""R2: 保存した証明で使われている規則の番号が、規則ファイルの名前でどれだけ覆われるか、名前が一意かを調べる。

usage: python r2_rule_names.py <problems.jsonl> <rules.txt> [<unabridged_rules.txt>]
"""
import collections
import json
import re
import sys

from vlma.actions import Action, parse_action

names = {}
for path in sys.argv[2:]:
    lines = [l.strip() for l in open(path) if l.strip()]
    for i, l in enumerate(lines):
        m = re.match(r"^(r\d+)\s+(.*)$", l)
        if m and i + 1 < len(lines) and "=>" in lines[i + 1]:
            names.setdefault(m[1], (m[2], path.split("/")[-1], lines[i + 1]))
used = collections.Counter()
for line in open(sys.argv[1]):
    c = json.loads(line)
    for a in c["actions"]:
        p = parse_action(a)
        if isinstance(p, Action) and p.kind == "apply":
            used[p.rule] += 1
total = sum(used.values())
print("used rule codes:", len(used), "steps:", total)
missing = {k: v for k, v in used.items() if k not in names}
print("without a name in the rule files:", missing)
by_name = collections.defaultdict(list)
for k, (n, _, _) in names.items():
    by_name[n].append(k)
print("duplicate names:", {n: ks for n, ks in by_name.items() if len(ks) > 1})
for k, v in used.most_common(40):
    if k in names:
        print(f"{k:6} {v:8}  {names[k][0]}   [{names[k][1]}]")
