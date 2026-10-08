"""R2: 保存した証明の各手が、目標の導出に必要か（目標から根拠をたどって届くか）を調べる。

usage: python r2_diag_cone.py <problems.jsonl> [--limit N]
各手の結論の番号と根拠の番号から、最後の手（目標）に向かう依存の木を後ろ向きにたどり、
たどり着かない手（目標に不要な手）の数を数える。
"""
import json
import re
import sys
from collections import Counter

from vlma.actions import Action, parse_action


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    unused_hist, total, with_unused, ar_unused = Counter(), 0, 0, 0
    n = 0
    for line in open(sys.argv[1]):
        c = json.loads(line)
        acts = [parse_action(a) for a in c["actions"]]
        steps = [a for a in acts if isinstance(a, Action) and a.kind == "apply"]
        by_id = {a.fact_id: a for a in steps}
        need, stack = set(), [steps[-1].fact_id]
        while stack:
            i = stack.pop()
            if i in need or i not in by_id:
                continue
            need.add(i)
            stack.extend(by_id[i].premises)
        unused = [a for a in steps if a.fact_id not in need]
        total += len(steps)
        unused_hist[min(len(unused), 5)] += 1
        with_unused += bool(unused)
        ar_unused += sum(a.rule == "AR" for a in unused)
        n += 1
        if limit and n >= limit:
            break
    print(json.dumps({"problems": n, "apply_steps": total,
                      "problems_with_unneeded_steps": with_unused,
                      "unneeded_steps_histogram(5=5以上)": dict(sorted(unused_hist.items())),
                      "unneeded_AR_steps": ar_unused}, ensure_ascii=False))


if __name__ == "__main__":
    main()
