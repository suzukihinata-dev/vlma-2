"""R2 段階4: 作成したデータセットの仕上げと整合性チェック。

usage: python r2_finalize.py --dir DIR --raw gen1.jsonl [--raw gen2.jsonl ...] [--verify-sample N] [--workers W]
1. problems.jsonl に fl_problem（前提の座標つき）が無ければ、生成機の出力から補う
2. 全問で: 問題 id が一意、全アクションが解析できる、図のファイルが全て存在する、ターン数が一致する
3. 図の寸法を無作為に抽出して確認する
4. 無作為に N 問を、保存したデータだけ（fl_problem・problem・actions・placement_seed）から検証器で再検証する
結果は DIR/finalize.json に書く。
"""
import hashlib
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

from PIL import Image


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def reverify(c):
    from vlma.steps import expand
    from vlma.verify import StepVerifier

    try:
        rec = {"fl_problem": c["fl_problem"], "llm_input_renamed": c["problem"]}
        v = StepVerifier(rec, seed=c["placement_seed"])
        verdicts = [v.check(s["output"]) for s in expand(c)]
        reached = [i + 1 for i, x in enumerate(verdicts) if x.goal_reached]
        return c["problem_id"], all(x.ok for x in verdicts) and reached == [len(verdicts)]
    except Exception as e:  # noqa: BLE001
        return c["problem_id"], f"{type(e).__name__}: {e}"[:150]


def main():
    d = Path(arg("--dir"))
    raws = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--raw"]
    n_sample = arg("--verify-sample", 2000, int)
    workers = arg("--workers", 20, int)
    rows = [json.loads(l) for l in open(d / "problems.jsonl") if l.strip()]
    out = {"problems": len(rows)}

    if any("fl_problem" not in r for r in rows):
        fl = {}
        for pth in raws:
            for line in open(pth):
                if line.strip():
                    f = json.loads(line)["fl_problem"]
                    fl[hashlib.sha1(f.encode()).hexdigest()[:12]] = f
        missing = 0
        for r in rows:
            if "fl_problem" not in r:
                if r["problem_id"] in fl:
                    r["fl_problem"] = fl[r["problem_id"]]
                else:
                    missing += 1
        out["fl_problem_missing"] = missing
        with open(d / "problems.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from vlma.actions import Action, parse_action
    from vlma.steps import expand, figure_path

    ids = [r["problem_id"] for r in rows]
    out["unique_ids"] = len(set(ids)) == len(ids)
    bad_actions = bad_turns = missing_figs = n_figs = 0
    for r in rows:
        acts = [parse_action(a) for a in r["actions"]]
        bad_actions += any(not isinstance(a, Action) for a in acts)
        bad_turns += len(acts) != r["n_turns"]
        n_con = sum(isinstance(a, Action) and a.kind == "construct" for a in acts)
        bad_turns += n_con + 1 != r["n_figures"]
        for k in range(r["n_figures"]):
            n_figs += 1
            missing_figs += not (d / figure_path(r["problem_id"], k)).exists()
    out.update({"bad_actions": bad_actions, "bad_turn_or_figure_counts": bad_turns,
                "figures_expected": n_figs, "figures_missing": missing_figs})

    random.seed(0)
    sizes = {}
    for r in random.sample(rows, min(1000, len(rows))):
        for k in range(r["n_figures"]):
            s = Image.open(d / figure_path(r["problem_id"], k)).size
            sizes[str(s)] = sizes.get(str(s), 0) + 1
    out["figure_sizes_sampled"] = sizes

    sample = random.sample(rows, min(n_sample, len(rows)))
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(reverify, sample, chunksize=8)
    failed = [x for x in res if x[1] is not True]
    out["reverified"] = len(res)
    out["reverify_failed"] = len(failed)
    out["reverify_failures"] = failed[:5]
    out["reverify_s"] = round(time.time() - t0, 1)
    json.dump(out, open(d / "finalize.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
