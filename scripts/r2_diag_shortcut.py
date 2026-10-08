"""R2: 検証器の抜け道の確認。作図のあとに目標の文をそのまま出すと、1手で受理され完了になるか。

usage: python r2_diag_shortcut.py <problems.jsonl> [--limit N]
"""
import json
import sys

from vlma.actions import Action, parse_action
from vlma.verify import StepVerifier


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 50
    n = accepted = reached = 0
    shown = 0
    for line in open(sys.argv[1]):
        c = json.loads(line)
        acts = [parse_action(a) for a in c["actions"]]
        rec = {"fl_problem": c["fl_problem"], "llm_input_renamed": c["problem"]}
        v = StepVerifier(rec, seed=c["placement_seed"])
        for a in acts:
            if isinstance(a, Action) and a.kind == "construct":
                v.check(a.format())
        pred, *args = c["goal"].split()
        jump = f"APPLY AR : {pred} {' '.join(args)} [900]"
        verdict = v.check(jump)
        n += 1
        accepted += verdict.ok
        reached += verdict.goal_reached
        if shown < 2:
            print(jump, "->", verdict)
            shown += 1
        if n >= limit:
            break
    print(json.dumps({"problems": n, "goal_stated_directly_accepted": accepted, "goal_reached": reached}))


if __name__ == "__main__":
    main()
