"""Hugging Face へのバックアップ用に、データを整える。図（PNG 約21万枚）は、ファイル数を減らすため、tar に束ねる。

usage: python r3_pack_for_hf.py --dir DIR --raw gen1.jsonl [--raw gen2.jsonl ...] --out OUT [--shard-mb 100] [--code code.tar.gz]
OUT の中身:
  README.md            データの説明（作り方、形式、件数、限界）
  problems.jsonl       コンパクトな形の問題（座標つきの前提、操作、図ごとの点の座標を含む）
  split.json, subset_1000.json, *_summary.json, stats.json, ...   分割・選定・件数の記録
  figs/shard-NNNN.tar  図（<問題id>/fig<k>.png）。figs/index.json が、問題 id → tar の対応
  raw/*.jsonl.gz       生成機の出力（再現と、問題 id → 乱数の種の対応のため）
  code-snapshot.tar.gz このデータを作り、読み込むコード（vlma、scripts、docs、設定）
"""
import gzip
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


README = """---
license: other
pretty_name: VLMA-2 geometry (R2)
tags: [geometry, theorem-proving, synthetic, genesisgeo]
---

# VLMA-2 geometry dataset (R2)

Synthetic plane-geometry problems with step-by-step proofs, generated with
[GenesisGeo](https://github.com/ZJUVAI/GenesisGeo) (commit `22fe601034ddc32b095c38d07a29ba051bdc8142`) and verified with its DDAR engine.
Private backup for the vlma-2 project. Each problem has a figure (rendered after every construction), a goal, and a proof written as one action per turn.

## Contents
- `problems.jsonl` — {n_problems} problems in a compact form. Per line: `problem_id`, `fl_problem` (premises with coordinates and the goal),
  `problem` (premise facts and goal as text), `goal`, `actions` (the construction and proof steps, with GenesisGeo rule numbers),
  `numerical_check` / `trivial` (facts the generator listed; not used as model input), `placement_seed` (which of the possible placements of
  auxiliary points matches the recorded facts), `figure_points` (point coordinates per figure), `n_turns`, `n_figures`.
  One-turn-per-sample training examples are built when loading (`vlma/steps.py: expand`); rule numbers become names (`vlma/rules.py`).
- `figs/` — figures as PNG (512x512), bundled in tar shards; `figs/index.json` maps problem id to shard. Path inside a shard: `<problem_id>/fig<k>.png`.
- `split.json` — train / val / test / excluded per problem. Problems that share a figure recipe (same construction sequence) are on the same side.
- `subset_1000.json` — the 1,000 problems (one per independent figure) used for the first training run.
- `raw/` — generator output (gzip). `code-snapshot.tar.gz` — code and docs that produced and read this data.

## Verification
Every step of every problem was replayed with the DDAR engine when the data was built. The final verifier (steps must follow from the cited facts, the
named rule must be able to derive the step, no shortcut to the goal) was re-run on about 2,500 sampled problems with no rejections; it was not re-run on all problems.

## Known limitations
- **Low diversity of figures.** The {n_problems} problems come from only about 2,159 independent random figures (generator seeds); one figure yields up to hundreds of problems.
  The first construction is one of four types for about 80% of the problems. Do not read the problem count as the number of distinct figures.
- Problems whose auxiliary point needs three or more conditions (about 9% of generated problems) are excluded: GenesisGeo's own translator does not support them. The data is biased toward easier problems.
- Figures show the problem at the time of each construction; the auxiliary-point placement is one valid choice among the possible ones.
- No negative examples are included.
"""


def main():
    d, out = Path(arg("--dir")), Path(arg("--out"))
    raws = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--raw"]
    shard_mb = arg("--shard-mb", 100, int)
    code = arg("--code", None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)

    n = 0
    for name in ("problems.jsonl", "split.json", "split_summary.json", "subset_1000.json", "subset_1000_summary.json",
                 "stats.json", "merge_manifest.json", "finalize.json", "figure_points.json", "turn_index.json"):
        if (d / name).exists():
            shutil.copy(d / name, out / name)
    for line in open(d / "problems.jsonl"):
        n += 1

    # 図を、tar に束ねる
    index, shard, size, k, tar = {}, 0, 0, 0, None
    limit = shard_mb * 2**20
    for line in open(d / "problems.jsonl"):
        pid = json.loads(line)["problem_id"]
        figs = sorted((d / "figs" / pid).glob("*.png"))
        if tar is None or size > limit:
            if tar:
                tar.close()
            shard += 1
            size = 0
            tar = tarfile.open(out / "figs" / f"shard-{shard:04d}.tar", "w")
        for p in figs:
            tar.add(p, arcname=f"{pid}/{p.name}")
            size += p.stat().st_size
        index[pid] = f"shard-{shard:04d}.tar"
        k += len(figs)
    if tar:
        tar.close()
    json.dump(index, open(out / "figs" / "index.json", "w"))

    for r in raws:
        dst = out / "raw" / (Path(r).parent.parent.name + "-" + Path(r).name + ".gz")
        with open(r, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
            shutil.copyfileobj(f, g)
    if code:
        shutil.copy(code, out / "code-snapshot.tar.gz")
    (out / "README.md").write_text(README.replace("{n_problems}", f"{n:,}"))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(json.dumps({"problems": n, "figures": k, "shards": shard, "files": files, "total_gb": round(total / 2**30, 2)}))


if __name__ == "__main__":
    main()
