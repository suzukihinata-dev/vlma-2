"""R2: 図が補助作図の提案に効いているかを、公開モデル GenesisGeo-2B で比べる。

usage: python r2_figure_ablation.py <in.jsonl> <out.json> [--limit N] [--k 8] [--base_url http://127.0.0.1:8000]
補助点が1つの問題だけを使い、補助点を除いた問題（前提＋目標）に対してモデルに補助点を提案させ、
DDAR で目標が証明できる割合を条件ごとに比べる。条件:
  figure   公式の描画の図（draw_clause_figure、512px）
  blank    同じ大きさの白い画像
  shuffled 別の問題の図
  text     画像なし（公式の qwen3_vl_text と同じ入力）
リクエストの作り方は GenesisGeo 公式（agent/vllm.py）の関数をそのまま使い、画像だけ差し替える。
"""
import base64
import io
import json
import re
import sys
import tempfile
import time

import requests
from PIL import Image
from transformers import AutoTokenizer

from newclid.agent.base import RESPONSE_PREFIX
from newclid.agent.vllm import (
    AUX_CANDIDATE_STOP, MAX_NEW_TOKENS, _build_vl_request_payload, _build_vl_text_request_payload,
)
from newclid.api import GeometricSolverBuilder
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions

CONDITIONS = ("figure", "blank", "shuffled", "text")


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def solves(premises, cons, goal, cache, key):
    if key in cache:
        return cache[key]
    text = "; ".join([premises, *([cons] if cons else [])]) + f" ? {goal}"
    try:
        solver = GeometricSolverBuilder(seed=123).load_problem_from_txt(text).build()
        ok = bool(solver.run(timeout=60))
    except Exception:  # noqa: BLE001
        ok = None  # 作図が構築できない
    cache[key] = ok
    return ok


def blank_url():
    buf = io.BytesIO()
    Image.new("RGB", (512, 512), "white").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    limit, K = arg("--limit", None, int), arg("--k", 8, int)
    base = arg("--base_url", "http://127.0.0.1:8000")
    model = requests.get(f"{base}/v1/models", timeout=60).json()["data"][0]["id"]
    stop_id = AutoTokenizer.from_pretrained(model, trust_remote_code=True).encode(
        AUX_CANDIDATE_STOP, add_special_tokens=False)[0]
    rows = [json.loads(l) for l in open(inp) if l.strip()]
    rows = [r for r in rows if r["llm_output_renamed"].count("x00") == 1][:limit]

    probs = []
    tmp = tempfile.mkdtemp()
    for i, rec in enumerate(rows):
        premises, goal = (s.strip() for s in rec["fl_problem"].split("?", 1))
        premises = premises.rstrip(";")
        builder = GeometricSolverBuilder(seed=123).load_problem_from_txt(f"{premises} ? {goal}")
        problem, defs = builder.problemJGEX, builder.defs
        fig_req = _build_vl_request_payload(
            mode="v1", depth=0, request_id=f"p{i}", problem=problem, aux_prefix="", defs=defs,
            root_problem_dsl=None, render_root=tmp, decoding_size=K)
        txt_req = _build_vl_text_request_payload(
            mode="v1", request_id=f"p{i}", problem=problem, aux_prefix="", defs=defs,
            root_problem_dsl=None, decoding_size=K)
        probs.append({"i": i, "premises": premises, "goal": goal, "fig": fig_req, "txt": txt_req})

    cache = {}
    for p in probs:  # 補助点なしで解ける問題は比較から除く
        p["no_aux"] = solves(p["premises"], "", p["goal"], cache, (p["i"], "")) is True
    use = [p for p in probs if not p["no_aux"]]
    blank = blank_url()

    def request(p, cond):
        base_req = p["txt"] if cond == "text" else p["fig"]
        messages = json.loads(json.dumps(base_req["messages"]))
        if cond in ("blank", "shuffled"):
            url = blank if cond == "blank" else use[(use.index(p) + 1) % len(use)]["fig"]["image_data_url"]
            messages[1]["content"][0]["image_url"]["url"] = url
        payload = {
            "model": model, "messages": messages, "continue_final_message": True,
            "add_generation_prompt": False, "max_tokens": MAX_NEW_TOKENS, "n": K,
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
            "stream": False, "stop": [AUX_CANDIDATE_STOP], "stop_token_ids": [stop_id],
            "include_stop_str_in_output": False, "seed": 1000 + p["i"],
        }
        r = requests.post(f"{base}/v1/chat/completions", json=payload, timeout=600)
        r.raise_for_status()
        out = []
        for c in r.json()["choices"]:
            cont = str(c["message"]["content"]).partition("</aux>")[0].rstrip()
            out.append(f"{RESPONSE_PREFIX} {base_req['new_point_name']} : {cont.lstrip()}")
        return out

    results = {c: [] for c in CONDITIONS}
    t0 = time.time()
    for cond in CONDITIONS:
        for p in use:
            cands = request(p, cond)
            rec = {"i": p["i"], "n": len(cands), "translated": 0, "built": 0, "solved": 0}
            for dsl in cands:
                body = dsl.replace("<aux>", "", 1).strip()
                cons = try_full_aux_dsl_to_constructions(body)
                if cons is None:
                    continue
                rec["translated"] += 1
                ok = solves(p["premises"], cons, p["goal"], cache, (p["i"], cons))
                if ok is None:
                    continue
                rec["built"] += 1
                rec["solved"] += bool(ok)
            results[cond].append(rec)
        print(cond, "done", round(time.time() - t0), "s", flush=True)

    def summarize(rs):
        n = sum(r["n"] for r in rs)
        return {
            "problems": len(rs), "samples": n,
            "translated_rate": round(sum(r["translated"] for r in rs) / n, 4),
            "built_rate": round(sum(r["built"] for r in rs) / n, 4),
            "solved_sample_rate": round(sum(r["solved"] for r in rs) / n, 4),
            "solved_at_k": sum(r["solved"] > 0 for r in rs),
            "solved_at_k_rate": round(sum(r["solved"] > 0 for r in rs) / len(rs), 4),
        }

    summary = {c: summarize(results[c]) for c in CONDITIONS}
    sol = {c: {r["i"]: r["solved"] > 0 for r in results[c]} for c in CONDITIONS}
    paired = {}
    for c in CONDITIONS[1:]:
        both = sum(sol["figure"][i] and sol[c][i] for i in sol["figure"])
        fig_only = sum(sol["figure"][i] and not sol[c][i] for i in sol["figure"])
        other_only = sum(not sol["figure"][i] and sol[c][i] for i in sol["figure"])
        paired[c] = {"both": both, "figure_only": fig_only, f"{c}_only": other_only}
    meta = {"k": K, "n_input_problems": len(rows), "excluded_solved_without_aux": len(probs) - len(use),
            "model": model}
    out = {"meta": meta, "summary": summary, "paired_solved_at_k": paired, "results": results}
    json.dump(out, open(outp, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"meta": meta, "summary": summary, "paired": paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
