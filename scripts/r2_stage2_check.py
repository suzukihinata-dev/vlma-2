"""R2 段階2 の確認: 1手ごとの形式への変換・逆変換・DDAR 検証を全件で行う。

usage: python r2_stage2_check.py <in.jsonl> <out.json> [--limit N]
1. to_steps → from_steps で元の llm_output_renamed に戻る（往復一致）
2. 正解の操作列を StepVerifier に1手ずつ入れ、全手が受理され、最後の手で完了になる
3. 壊した出力が、期待する理由で却下される（検証器の確認。学習用の負例ではない）
"""
import json
import sys
import time
import traceback

from vlma.actions import Action, BUILD_FAILED, INVALID_STATEMENT, NOT_HOLD, PARSE_FAILED, parse_action
from vlma.steps import from_steps, to_steps
from vlma.verify import StepVerifier


def norm(s):
    return " ".join(s.split())


def probes(rec, steps):
    """壊した出力と、期待する却下の理由（None は「理由は問わないが受理されたら記録」）。"""
    applies = [parse_action(s["output"]) for s in steps]
    applies = [a for a in applies if isinstance(a, Action) and a.kind == "apply"]
    first = applies[0]
    out = [
        ("garbage", "hello world", PARSE_FAILED),
        ("two_steps_in_one", steps[-1]["output"] + "\n" + steps[-1]["output"], PARSE_FAILED),
        ("unknown_predicate", "CONSTRUCT z : foo a b z [900]", BUILD_FAILED),
        ("unknown_point", f"APPLY r01 : {first.pred} " + " ".join(["q9"] * len(first.args)) + " [900]", INVALID_STATEMENT),
    ]
    # 結論の点を1つ取り替えた手。偶然成り立つこともあるので、分布だけ記録する。
    last = applies[-1]
    pts = sorted({p for a in applies for p in a.args})
    for k in range(len(last.args)):
        alt = next((p for p in pts if p != last.args[k]), None)
        if alt is None:
            continue
        args = list(last.args)
        args[k] = alt
        mutated = Action(kind="apply", rule=last.rule, pred=last.pred, args=tuple(args),
                         fact_id=last.fact_id, premises=last.premises).format()
        out.append((f"mutated_arg{k}", mutated, None))
    return out


def check(rec):
    res = {}
    steps = to_steps(rec)
    res["n_turns"] = len(steps)
    res["roundtrip"] = from_steps(steps) == norm(rec["llm_output_renamed"])
    v = StepVerifier(rec)
    t0 = time.time()
    verdicts = [v.check(s["output"]) for s in steps]
    res["verify_s"] = round(time.time() - t0, 3)
    res["all_accepted"] = all(x.ok for x in verdicts)
    res["rejected_at"] = [i + 1 for i, x in enumerate(verdicts) if not x.ok][:3]
    reached = [i + 1 for i, x in enumerate(verdicts) if x.goal_reached]
    res["goal_reached_turns"] = reached
    res["goal_only_at_last"] = reached == [len(steps)]
    # 壊した出力: 正解の作図をすべて受理した状態（＝最後の手の直前）で試す
    pr = {}
    base = StepVerifier(rec)
    for s in steps[:-1]:
        base.check(s["output"])
    for name, text, expected in probes(rec, steps):
        verdict = base.check(text)
        pr[name] = {"ok": verdict.ok, "reason": verdict.reason, "expected": expected}
    res["probes"] = pr
    return res


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rows = [json.loads(l) for l in open(inp) if l.strip()][:limit]
    results = []
    for i, rec in enumerate(rows):
        try:
            r = check(rec)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:]}
        r["i"] = i
        results.append(r)
    ok = [r for r in results if "error" not in r]
    strict = [(r["probes"][n]) for r in ok for n in r["probes"] if r["probes"][n]["expected"]]
    mut = [r["probes"][n] for r in ok for n in r["probes"] if n.startswith("mutated")]
    reasons = {}
    for m in mut:
        key = "accepted(偶然成立)" if m["ok"] else m["reason"]
        reasons[key] = reasons.get(key, 0) + 1
    summary = {
        "n": len(results),
        "errors": len(results) - len(ok),
        "turns_total": sum(r["n_turns"] for r in ok),
        "roundtrip": sum(r["roundtrip"] for r in ok),
        "all_accepted": sum(r["all_accepted"] for r in ok),
        "goal_only_at_last": sum(r["goal_only_at_last"] for r in ok),
        "strict_probes": len(strict),
        "strict_probes_correct": sum((not p["ok"]) and p["reason"] == p["expected"] for p in strict),
        "mutated_probes": len(mut),
        "mutated_by_reason": reasons,
        "max_verify_s": max((r["verify_s"] for r in ok), default=None),
    }
    json.dump({"summary": summary, "results": results}, open(outp, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
