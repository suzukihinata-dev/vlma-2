"""R2: 保存した証明（参照）が、新しい検証器（根拠から導けるか）で全手受理されるかを調べる。

usage: python r2_diag_reference.py <problems.jsonl> [--limit N] [--skip M]
"""
import collections
import json
import sys

from vlma.actions import Action, parse_action
from vlma.reward import step_rewards
from vlma.steps import expand
from vlma.verify import StepVerifier


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 100
    skip = int(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 0
    ok_all = n = 0
    reasons, rules_bad, kinds = collections.Counter(), collections.Counter(), collections.Counter()
    steps_total = steps_bad = 0
    ex = []
    first = collections.Counter()  # 各問題で最初に却下された手（後続の手は、その番号を根拠にして連鎖的に却下される）
    first_ex = []
    for i, line in enumerate(open(sys.argv[1])):
        if i < skip:
            continue
        c = json.loads(line)
        samples = expand(c)
        outs = [s["output"] for s in samples]
        rec = {"fl_problem": c["fl_problem"], "llm_input_renamed": c["problem"]}
        v = StepVerifier(rec, seed=c["placement_seed"], reference=outs)
        verdicts = [v.check(o) for o in outs]
        bad = [(k, x) for k, x in enumerate(verdicts) if not x.ok]
        if bad:
            k0, x0 = bad[0]
            a0 = parse_action(outs[k0])
            first[(x0.reason, a0.rule if a0.kind == "apply" else "construct", a0.pred if a0.kind == "apply" else "-")] += 1
            if len(first_ex) < 4:
                missing = [p for p in a0.premises if p not in v._facts] if a0.kind == "apply" else []
                first_ex.append((x0.reason, outs[k0][:110], "missing cited ids:", missing,
                                 "numeric ids dropped:", sorted(set(i for f in c["numerical_check"] + c["trivial"] for i in __import__("re").findall(r"\\[(\\d+)\\]", f)))[:6]))
        steps_total += len(verdicts)
        steps_bad += len(bad)
        for k, x in bad:
            reasons[x.reason] += 1
            a = parse_action(outs[k])
            rules_bad[a.rule if a.kind == "apply" else "construct"] += 1
            if len(ex) < 3:
                ex.append((x.reason, outs[k][:100]))
        n_ref = sum(o.startswith("APPLY") for o in outs)
        kinds[step_rewards(verdicts, n_ref)[1]] += 1
        ok_all += not bad
        n += 1
        if n >= limit:
            break
    print(json.dumps({"problems": n, "all_steps_accepted": ok_all, "steps": steps_total, "steps_rejected": steps_bad,
                      "reasons": dict(reasons), "trajectory_kinds_of_reference": dict(kinds),
                      "rejected_by_rule(top)": dict(rules_bad.most_common(6))}, ensure_ascii=False))
    print("examples:", ex)
    print("FIRST rejection per failed problem:", {" / ".join(k): v_ for k, v_ in first.most_common(8)})
    for e in first_ex:
        print("first-failure example:", e)


if __name__ == "__main__":
    main()
