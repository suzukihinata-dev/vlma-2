"""R2 段階4: not_hold で却下された問題を詳しく調べる。

usage: python r2_diag_reject2.py <rejected.jsonl> [--limit N]
却下された手の直前まで検証器で再生し、(1) その手の結論が数値的に成立するか、
(2) 記録された numerical_check / trivial の事実が、再構築した図で数値的に成立するか、
(3) 補助点の位置を変えて再構築すると結果が変わるか、を調べる。
"""
import json
import sys
from collections import Counter

from newclid.discovery.extraction import parsing
from newclid.statement import Statement
from vlma.actions import parse_action
from vlma.steps import to_steps
from vlma.verify import StepVerifier


def numeric(proof, tokens):
    try:
        st = Statement.from_tokens(tokens, proof.dep_graph)
        return None if st is None else bool(st.check_numerical())
    except Exception:  # noqa: BLE001
        return None


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    rows = [r for r in rows if r["reason"] == "not_hold"]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    tally = Counter()
    for k, r in enumerate(rows[:limit]):
        rec = r["record"]
        steps = to_steps(rec)
        out = {}
        for seed in (123, 7, 99):
            v = StepVerifier(rec, seed=seed, engine="both")
            bad = None
            for i, s in enumerate(steps):
                if not v.check(s["output"]).ok:
                    bad = i
                    break
            out[seed] = bad
            if seed == 123 and bad is not None:
                proof = v._saturated_proof("c")
                a = parse_action(steps[bad]["output"])
                out["stmt_numeric"] = numeric(proof, a.conclusion_tokens)
                facts = parsing.parse_fact_segments(
                    parsing.extract_tag_content(rec["llm_output_renamed"], "numerical_check")
                    + " ; " + parsing.extract_tag_content(rec["llm_output_renamed"], "trivial"))
                bad_facts = [f"{p} {' '.join(a_)}" for p, a_, _ in facts if numeric(proof, (p,) + tuple(a_)) is False]
                out["numerical_facts_bad"] = bad_facts[:3]
                out["n_facts"] = len(facts)
        tally[(out[123] is not None, out[7] is None or out[99] is None)] += 1
        if k < 8:
            print(json.dumps({"i": k, "turn": r["turn"], "action": r["action"], **{str(a): b for a, b in out.items()}},
                             ensure_ascii=False), flush=True)
    print("tally (fails at seed123, passes at some other seed):", dict(tally))


if __name__ == "__main__":
    main()
