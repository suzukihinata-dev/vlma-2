"""R2: 全問で、規則の番号が名前に変わり、元の番号に戻ることを確認する。

usage: python r2_check_names_all.py <problems.jsonl> [--workers W]
各問題を expand（番号 → 名前）し、手ごとに: 名前が表にあり結論の述語に合う、名前から番号に戻すと元の番号と一致する、
出力に番号（r72 など）や AR・Transfer の語が規則として残っていない、を確認する。
"""
import collections
import json
import multiprocessing as mp
import re
import sys

from vlma import rules
from vlma.actions import Action, parse_action
from vlma.steps import expand

NUM = re.compile(r"^APPLY (r\d+|AR|Transfer) ")


def check(line):
    c = json.loads(line)
    c.pop("figure_points", None)  # 規則名の変換だけを確かめる（情報欄は使わない）
    out = collections.Counter()
    try:
        steps = expand(c)
        assert len(steps) == len(c["actions"])
        for s, raw in zip(steps, c["actions"]):
            a, orig = parse_action(s["output"]), parse_action(raw)
            if a.kind == "construct":
                assert s["output"] == raw
                continue
            assert not NUM.match(s["output"]), s["output"]
            assert rules.is_valid(a.rule, a.pred), (a.rule, a.pred)
            assert rules.rule_code(a.rule) == orig.rule, (a.rule, orig.rule)
            assert a.pred == orig.pred and a.args == orig.args and a.fact_id == orig.fact_id
            assert a.to_genesis().split("] ")[1].split()[0] == orig.rule  # 番号に戻る
            out[a.rule] += 1
        return out, None
    except Exception as e:  # noqa: BLE001
        return out, f"{c['problem_id']}: {type(e).__name__}: {e}"[:200]


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    total, errors, n = collections.Counter(), [], 0
    with open(sys.argv[1]) as f, mp.get_context("fork").Pool(workers) as pool:
        for out, err in pool.imap_unordered(check, f, chunksize=64):
            total.update(out)
            n += 1
            if err:
                errors.append(err)
    print(json.dumps({"problems": n, "problems_with_errors": len(errors), "steps_converted": sum(total.values()),
                      "distinct_names_used": len(total), "first_errors": errors[:3]}, ensure_ascii=False))
    for k, v in total.most_common(5):
        print(f"  {k} {v}")


if __name__ == "__main__":
    main()
