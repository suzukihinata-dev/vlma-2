"""R2 段階3 の確認: 作図ごとの図を描き、数値の整合性と画像の健全性を調べる。

usage: python r2_stage3_check.py <in.jsonl> <out_dir> [--limit N]
各問題で、作図の数 + 1 枚の図を描き、次を確認する。
1. 画像が存在し 512px 幅で、真っ白ではない
2. 隣り合う図が違う（作図で点が増えるので画像は変わる）
3. 最後の図の数値の状態で、全手の結論・目標・情報（numerical_check, trivial）が数値的に成立する
   （図の座標が操作列と矛盾しない）
"""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from newclid.discovery.extraction import parsing
from newclid.statement import Statement
from vlma.actions import Action
from vlma.figures import find_placement_seed, render_problem_figures
from vlma.steps import extract_actions


def numeric_failures(proof, rec):
    dg = proof.dep_graph
    toks = []
    for a in extract_actions(rec):
        if isinstance(a, Action) and a.kind == "apply":
            toks.append(("step", a.conclusion_tokens))
    g = parsing.extract_goal(rec["llm_input_renamed"])
    toks.append(("goal", (g.predicate,) + tuple(g.args)))
    out = rec["llm_output_renamed"]
    for tag in ("numerical_check", "trivial"):
        for pred, args, _ in parsing.parse_fact_segments(parsing.extract_tag_content(out, tag)):
            toks.append((tag, (pred,) + tuple(args)))
    bad = []
    for kind, t in toks:
        try:
            st = Statement.from_tokens(t, dg)
            ok = st is not None and bool(st.check_numerical())
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            bad.append((kind, " ".join(t)))
    return bad, len(toks)


def check(rec, root):
    seed = find_placement_seed(rec)
    figs = render_problem_figures(rec, root, seed or 0)
    res = {"n_figs": len(figs), "placement_seed": seed}
    hashes, sizes, blank = [], set(), 0
    for rel, _ in figs:
        p = Path(root) / rel
        im = Image.open(p).convert("L")
        sizes.add(im.size)
        lo, hi = im.getextrema()
        blank += (lo == hi)
        hashes.append(hashlib.sha1(p.read_bytes()).hexdigest())
    res["sizes"] = sorted(sizes)
    res["blank"] = blank
    res["adjacent_differ"] = all(a != b for a, b in zip(hashes, hashes[1:]))
    bad, n = numeric_failures(figs[-1][1], rec)
    res["n_checked"] = n
    res["numeric_bad"] = bad[:6]
    res["n_numeric_bad"] = len(bad)
    return res


def main():
    inp, root = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    rows = [json.loads(l) for l in open(inp) if l.strip()][:limit]
    results, t0 = [], time.time()
    for i, rec in enumerate(rows):
        try:
            r = check(rec, root)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-500:]}
        r["i"] = i
        results.append(r)
    ok = [r for r in results if "error" not in r]
    summary = {
        "n": len(results),
        "errors": len(results) - len(ok),
        "figures": sum(r["n_figs"] for r in ok),
        "blank_images": sum(r["blank"] for r in ok),
        "sizes": sorted({tuple(s) for r in ok for s in r["sizes"]}),
        "adjacent_differ": sum(r["adjacent_differ"] for r in ok),
        "problems_all_numeric_ok": sum(r["n_numeric_bad"] == 0 for r in ok),
        "facts_checked": sum(r["n_checked"] for r in ok),
        "facts_numeric_bad": sum(r["n_numeric_bad"] for r in ok),
        "seconds": round(time.time() - t0, 1),
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(Path(root) / "stage3-check.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
