"""R2 段階1 probe: 生成データの補助作図と証明の手を DDAR に再生できるか調べる。

usage: python r2_replay_probe.py <in.jsonl> <out.json> [--limit N] [--seed S] [--only i,j,...] [--saturate]
各レコードについて、<aux> を作図に変換して問題に足し、DDAR を実行して
(1) 目標が証明されるか (2) 各手の結論が DDAR の状態で成立するか を記録する。
"""
import json
import sys
import time
import traceback

from newclid.api import GeometricSolverBuilder
from newclid.discovery.extraction import parsing
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.statement import Statement


def split_steps(proof_text):
    steps = []
    for raw in parsing.strip_tags(proof_text).split(";"):
        if not raw.strip():
            continue
        parsed = parsing.parse_proof_step(raw)
        steps.append(parsed)  # None のまま残す（解析失敗を数えるため）
    return steps


def probe(rec, seed=123, saturate=False):
    out = rec["llm_output_renamed"]
    res = {"seed": rec.get("seed")}
    aux_body = parsing.extract_tag_content(out, "aux").strip()
    aux_cons = try_full_aux_dsl_to_constructions(aux_body) if aux_body else ""
    res["aux_translated"] = aux_cons is not None
    if aux_cons is None:
        return res
    fl = rec["fl_problem"]
    prem, goal = fl.split("?", 1)
    prem = prem.strip().rstrip(";")
    text = f"{prem}; {aux_cons} ?{goal}" if aux_cons else fl
    t0 = time.time()
    builder = GeometricSolverBuilder(seed=seed).load_problem_from_txt(text)
    if saturate:
        # 目標を外して飽和まで推論させる。目標の成否は後で Statement で確かめる。
        builder = builder.del_goals()
    solver = builder.build()
    solved_run = bool(solver.run(timeout=120))
    res["solved"] = solved_run if not saturate else None
    res["elapsed_s"] = round(time.time() - t0, 3)
    if saturate:
        gp, ga = parsing.extract_goal(rec["llm_input_renamed"]).predicate, parsing.extract_goal(rec["llm_input_renamed"]).args
        gst = Statement.from_tokens((gp,) + tuple(ga), solver.proof.dep_graph)
        res["solved"] = bool(gst is not None and gst.check())
    steps = split_steps(parsing.extract_tag_content(out, "proof"))
    res["n_steps"] = len(steps)
    res["n_parse_fail"] = sum(s is None for s in steps)
    held, bad = 0, []
    dg = solver.proof.dep_graph
    for s in steps:
        if s is None:
            continue
        try:
            st = Statement.from_tokens((s[0],) + tuple(s[1]), dg)
            ok = st is not None and st.check()
        except Exception as e:  # noqa: BLE001
            ok = False
            bad.append(f"{s[0]} {' '.join(s[1])}: {type(e).__name__}")
            continue
        if ok:
            held += 1
        else:
            num = None if st is None else bool(st.check_numerical())
            bad.append(f"{s[0]} {' '.join(s[1])} (statement={'None' if st is None else 'ok'}, numerical={num})")
    res["n_steps_hold"] = held
    res["steps_not_hold"] = bad[:5]
    return res


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 123
    saturate = "--saturate" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = {int(x) for x in sys.argv[sys.argv.index("--only") + 1].split(",")}
    rows = [json.loads(l) for l in open(inp) if l.strip()]
    rows = rows[:limit]
    results = []
    for i, rec in enumerate(rows):
        if only is not None and i not in only:
            continue
        try:
            r = probe(rec, seed, saturate)
        except Exception as e:  # noqa: BLE001
            r = {"seed": rec.get("seed"), "error": f"{type(e).__name__}: {e}",
                 "trace": traceback.format_exc()[-400:]}
        r["i"] = i
        results.append(r)
    ok = [r for r in results if "error" not in r]
    summary = {
        "n": len(results),
        "errors": sum("error" in r for r in results),
        "aux_translated": sum(r.get("aux_translated", False) for r in ok),
        "solved": sum(r.get("solved", False) for r in ok),
        "all_steps_hold": sum(r.get("n_steps_hold") == r.get("n_steps") and r.get("n_parse_fail") == 0 for r in ok),
        "steps_total": sum(r.get("n_steps", 0) for r in ok),
        "steps_hold": sum(r.get("n_steps_hold", 0) for r in ok),
    }
    json.dump({"summary": summary, "results": results}, open(outp, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
