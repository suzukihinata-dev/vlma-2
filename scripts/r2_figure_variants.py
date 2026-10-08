"""R2: 図の描き方の変種を GenesisGeo-2B で比べる（r2_figure_ablation.py の続き）。

usage: python r2_figure_variants.py <in.jsonl> <out.json> --variants a,b,c [--limit N] [--k 8] [--preview DIR]
変種ごとに「その問題の図」と「別の問題の図」で補助点を提案させ、DDAR で目標が証明できた割合を比べる。
両者の差が、問題固有の情報をモデルが使えている度合いの目安になる。
変種: official（公式の描画）, annot_off（印なし）, lower（ラベルを小文字に）, big（ラベルを大きく）,
      lower_big（小文字で大きく）, zoom（点の範囲に寄せる）, zoom_lower_big。公式以外は、モデルの学習時と違う見た目になる点に注意。
"""
import base64
import io
import json
import os
import sys
import tempfile
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from transformers import AutoTokenizer

from newclid.agent.base import RESPONSE_PREFIX
from newclid.agent.vllm import AUX_CANDIDATE_STOP, MAX_NEW_TOKENS, _build_vl_request_payload
from newclid.api import GeometricSolverBuilder
from newclid.evaluation.search_runtime import build_problem_proof, try_full_aux_dsl_to_constructions
from newclid.generation.writer import save_figure_as_png
from newclid.numerical.draw_clause_figure import draw_clause_figure

from r2_figure_ablation import arg, solves  # 同じディレクトリに置く


def render_variant(problem, defs, variant, path):
    proof = build_problem_proof(problem, defs)
    fig = draw_clause_figure(proof, problem, None, proof.rng,
                             draw_annotations=(variant != "annot_off"), theme=None)
    ax = fig.axes[0]
    for t in ax.texts:
        if "lower" in variant:
            t.set_text(t.get_text().lower())
        if "big" in variant:
            t.set_fontsize(t.get_fontsize() * 1.6)
    if "zoom" in variant:
        # 点の範囲に描画領域を寄せる（大きな円などが範囲を決めて、点が小さくなるのを防ぐ）
        pts = [n.num for n in proof.symbols_graph.name2node.values() if hasattr(n.num, "x")]
        xs, ys = [q.x for q in pts], [q.y for q in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.35 or 1.0
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
    save_figure_as_png(fig, png_path=str(path), img_pixels=512, direct_png=True)
    plt.close(fig)
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    variants = arg("--variants", "official").split(",")
    limit, K = arg("--limit", None, int), arg("--k", 8, int)
    base = arg("--base_url", "http://127.0.0.1:8000")
    preview = arg("--preview", None)
    if preview:
        os.makedirs(preview, exist_ok=True)
    model = requests.get(f"{base}/v1/models", timeout=60).json()["data"][0]["id"]
    stop_id = AutoTokenizer.from_pretrained(model, trust_remote_code=True).encode(
        AUX_CANDIDATE_STOP, add_special_tokens=False)[0]
    rows = [json.loads(l) for l in open(inp) if l.strip()]
    rows = [r for r in rows if r["llm_output_renamed"].count("x00") == 1][:limit]
    tmp = tempfile.mkdtemp()
    probs = []
    for i, rec in enumerate(rows):
        premises, goal = (s.strip() for s in rec["fl_problem"].split("?", 1))
        premises = premises.rstrip(";")
        b = GeometricSolverBuilder(seed=123).load_problem_from_txt(f"{premises} ? {goal}")
        req = _build_vl_request_payload(mode="v1", depth=0, request_id=f"p{i}", problem=b.problemJGEX,
                                        aux_prefix="", defs=b.defs, root_problem_dsl=None,
                                        render_root=tmp, decoding_size=K)
        probs.append({"i": i, "premises": premises, "goal": goal, "req": req,
                      "problem": b.problemJGEX, "defs": b.defs})
    cache = {}
    for p in probs:
        p["no_aux"] = solves(p["premises"], "", p["goal"], cache, (p["i"], "")) is True
    use = [p for p in probs if not p["no_aux"]]

    def propose(p, url):
        messages = json.loads(json.dumps(p["req"]["messages"]))
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
        return [f"{RESPONSE_PREFIX} {p['req']['new_point_name']} : "
                + str(c["message"]["content"]).partition("</aux>")[0].rstrip().lstrip()
                for c in r.json()["choices"]]

    def score(p, dsls):
        rec = {"i": p["i"], "n": len(dsls), "built": 0, "solved": 0}
        for dsl in dsls:
            cons = try_full_aux_dsl_to_constructions(dsl.replace("<aux>", "", 1).strip())
            if cons is None:
                continue
            ok = solves(p["premises"], cons, p["goal"], cache, (p["i"], cons))
            if ok is None:
                continue
            rec["built"] += 1
            rec["solved"] += bool(ok)
        return rec

    out, t0 = {}, time.time()
    for v in variants:
        urls = {}
        for p in use:
            path = f"{preview}/{v}_{p['i']}.png" if preview else f"{tmp}/{v}_{p['i']}.png"
            urls[p["i"]] = render_variant(p["problem"], p["defs"], v, path)
        if preview and limit and limit <= 3:
            continue  # 見た目の確認だけ
        res = {"matched": [], "shuffled": []}
        for k, p in enumerate(use):
            res["matched"].append(score(p, propose(p, urls[p["i"]])))
            other = use[(k + 1) % len(use)]
            res["shuffled"].append(score(p, propose(p, urls[other["i"]])))
        n = sum(r["n"] for r in res["matched"])
        summ = {c: {"sample_rate": round(sum(r["solved"] for r in res[c]) / n, 4),
                    "solved_at_k": sum(r["solved"] > 0 for r in res[c]),
                    "built_rate": round(sum(r["built"] for r in res[c]) / n, 4)} for c in res}
        m = {r["i"]: r["solved"] for r in res["matched"]}
        s = {r["i"]: r["solved"] for r in res["shuffled"]}
        diffs = [(m[i] - s[i]) / K for i in m]
        summ["gap_matched_minus_shuffled"] = round(sum(diffs) / len(diffs), 4)
        summ["matched_only"] = sum(m[i] > 0 and s[i] == 0 for i in m)
        summ["shuffled_only"] = sum(s[i] > 0 and m[i] == 0 for i in m)
        out[v] = {"summary": summ, "results": res}
        print(v, json.dumps(summ, ensure_ascii=False), round(time.time() - t0), "s", flush=True)
    json.dump({"k": K, "n_problems": len(use), "variants": out}, open(outp, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
