"""R2: 保存した証明で、最初に invalid_citation になる手の詳細（どの番号が、なぜ存在しないか）を出す。

usage: python r2_diag_citation.py <problems.jsonl> [--limit N] [--skip M]
"""
import json
import sys

from vlma.actions import parse_action
from vlma.steps import expand
from vlma.verify import StepVerifier


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 80
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    shown = n = 0
    for i, line in enumerate(open(sys.argv[1])):
        if i < skip:
            continue
        n += 1
        if n > limit or shown >= 3:
            break
        c = json.loads(line)
        outs = [s["output"] for s in expand(c)]
        v = StepVerifier({"fl_problem": c["fl_problem"], "llm_input_renamed": c["problem"]}, seed=c["placement_seed"], reference=outs)
        for k, o in enumerate(outs):
            x = v.check(o)
            if not x.ok:
                if x.reason == "invalid_citation":
                    a = parse_action(o)
                    missing = [p for p in a.premises if p not in v._facts]
                    print("PROBLEM:", c["problem"][:300])
                    print("ACTIONS before:", [t[:60] for t in outs[max(0, k - 2):k]])
                    print("REJECTED:", o[:160], "| missing:", missing, "| known ids:", sorted(v._facts)[:40])
                    print("numeric/trivial facts:", (c["numerical_check"] + c["trivial"])[:4])
                    print("-----")
                    shown += 1
                break


if __name__ == "__main__":
    main()
