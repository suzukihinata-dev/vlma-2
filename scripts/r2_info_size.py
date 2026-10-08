"""R2: 情報欄（適用した手に関わる点の長さと角度）の、サンプルごとの長さを調べる。

usage: python r2_info_size.py <problems.jsonl> [--limit N]
"""
import json
import sys

from vlma.steps import expand

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 2000
chars, last = [], []
shown = False
n = 0
for line in open(sys.argv[1]):
    c = json.loads(line)
    steps = expand(c)
    chars.extend(len(s["info"]) for s in steps)
    last.append(len(steps[-1]["info"]))
    if not shown and len(steps) > 8:
        print("例（turn 9 の情報欄）:", steps[8]["info"][:400])
        print("  turn 9 までの出力:", steps[7]["output"])
        shown = True
    n += 1
    if n >= limit:
        break
chars.sort(); last.sort()
q = lambda a, p: a[int(p * (len(a) - 1))]
print(json.dumps({"problems": n, "samples": len(chars),
                  "info_chars_median": q(chars, .5), "info_chars_p90": q(chars, .9), "info_chars_max": chars[-1],
                  "final_turn_info_chars_median": q(last, .5), "final_turn_info_chars_p90": q(last, .9),
                  "final_turn_info_chars_max": last[-1]}))
