"""R2: 作成済みのデータセットに、図ごとの点の座標（figure_points）を追加する。

usage: python r2_add_figure_points.py --dir DIR [--workers W]
各問題を、保存した placement_seed で組み立て直し、作図の数 + 1 個の状態の座標を保存する。
元の problems.jsonl は problems.before-figure-points.jsonl に残す。
"""
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def points_of(c):
    from newclid.api import GeometricSolverBuilder
    from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
    from vlma.actions import Action, parse_action
    from vlma.figures import figure_points

    try:
        acts = [parse_action(a) for a in c["actions"]]
        cons = [try_full_aux_dsl_to_constructions(a.to_genesis()) for a in acts if isinstance(a, Action) and a.kind == "construct"]
        prem, goal = c["fl_problem"].split("?", 1)
        out = []
        for k in range(len(cons) + 1):
            text = "; ".join([prem.strip().rstrip(";"), *cons[:k]]) + f" ? {goal.strip()}"
            proof = GeometricSolverBuilder(seed=c["placement_seed"]).load_problem_from_txt(text).build(max_attempts=100).proof
            out.append(figure_points(proof))
        return c["problem_id"], out
    except Exception as e:  # noqa: BLE001
        return c["problem_id"], f"{type(e).__name__}: {e}"[:150]


def main():
    d = Path(arg("--dir"))
    workers = arg("--workers", 22, int)
    rows = [json.loads(l) for l in open(d / "problems.jsonl") if l.strip()]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = dict(pool.imap_unordered(points_of, rows, chunksize=16))
    failed = {k: v for k, v in res.items() if isinstance(v, str)}
    shutil.copy(d / "problems.jsonl", d / "problems.before-figure-points.jsonl")
    n_ok = 0
    with open(d / "problems.jsonl", "w") as f:
        for r in rows:
            v = res[r["problem_id"]]
            if isinstance(v, str):
                continue  # 座標が取れなかった問題は外す（件数は報告する）
            r["figure_points"] = v
            n_ok += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {"problems_in": len(rows), "with_points": n_ok, "failed": len(failed),
           "first_failures": list(failed.items())[:5], "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(d / "figure_points.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
