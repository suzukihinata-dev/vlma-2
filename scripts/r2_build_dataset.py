"""R2 段階4: 生成データから、検証済みの学習用データセットを作る。

usage: python r2_build_dataset.py --out DIR <in.jsonl> [<in2.jsonl> ...] [--limit N] [--workers W]
各問題について次を行い、通った問題だけを出力する。
  1. 重複の除去（fl_problem のハッシュ）
  1b. 補助点の置き場所（seed）を、記録された数値の事実が成り立つように決める。図と検証で同じ seed を使う
  2. 全手を StepVerifier で検証（全手が受理され、完了が最後の手だけ）
  3. 作図ごとの図を描く（図が描けなかった問題は除く）
出力: DIR/problems.jsonl（コンパクトな形。手ごとのサンプルは vlma.steps.expand で作る）、
      DIR/figs/<問題id>/fig<k>.png、DIR/rejected.jsonl（却下された問題と理由）、DIR/stats.json（件数・理由別の除外・時間・容量）
--limit は、検証を通った問題がこの件数に達したら打ち切る（生成機が要求より多く出す問題への対処）。
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

OUT = None


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def process(rec):
    """1問を処理して (状態, コンパクトな形 or 理由, 秒数) を返す。通らなかった問題は元の記録を添える。"""
    status, payload, t = _process(rec)
    if status not in ("ok", "rejected_verify"):
        payload = {"detail": payload, "record": rec}
    return status, payload, t


def _process(rec):
    from vlma.figures import figure_points, find_placement_seed, render_problem_figures
    from vlma.steps import expand, to_compact
    from vlma.verify import StepVerifier

    t = {}
    try:
        t0 = time.time()
        try:
            compact = to_compact(rec)
        except ValueError as e:
            if "unsupported aux segment" in str(e):  # 1つの補助作図が複数の点を同時に出す形式
                return "rejected_unsupported", str(e)[:120], t
            raise
        samples = expand(compact)
        t["convert"] = time.time() - t0
        t0 = time.time()
        try:
            seed = find_placement_seed(rec)
        except ValueError as e:
            if "not translatable" in str(e):  # 補助点の条件が3つ以上など、GenesisGeo の変換関数が対応しない作図
                return "rejected_unsupported", str(e)[:120], t
            raise
        t["placement"] = time.time() - t0
        if seed is None:
            return "rejected_placement", "no seed builds the construction and satisfies the recorded facts", t
        t0 = time.time()
        v = StepVerifier(rec, seed=seed)
        verdicts = [v.check(s["output"]) for s in samples]
        t["verify"] = time.time() - t0
        if not all(x.ok for x in verdicts):
            bad = next(i for i, x in enumerate(verdicts) if not x.ok)
            return "rejected_verify", {"turn": bad + 1, "reason": verdicts[bad].reason,
                                       "detail": verdicts[bad].detail, "action": samples[bad]["output"],
                                       "n_turns": len(samples), "record": rec}, t
        reached = [i + 1 for i, x in enumerate(verdicts) if x.goal_reached]
        if reached != [len(samples)]:
            return "rejected_goal", f"goal reached at {reached}", t
        t0 = time.time()
        figs = render_problem_figures(rec, OUT, seed)
        t["figures"] = time.time() - t0
        compact["placement_seed"] = seed
        compact["figure_points"] = [figure_points(p) for _, p in figs]  # 情報欄（図の座標から計算）の元データ
        compact["n_turns"] = len(samples)
        compact["n_figures"] = len(figs)
        return "ok", compact, t
    except Exception as e:  # noqa: BLE001
        return "error", f"{type(e).__name__}: {e}"[:200] + " | " + traceback.format_exc()[-200:], t


def init(out):
    global OUT
    OUT = out


def main():
    paths = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    out = Path(arg("--out"))
    limit = arg("--limit", None, int)
    workers = arg("--workers", 16, int)
    out.mkdir(parents=True, exist_ok=True)

    seen, records, duplicates, total_in = set(), [], 0, 0
    for pth in paths:
        for line in open(pth):
            if not line.strip():
                continue
            total_in += 1
            rec = json.loads(line)
            key = hashlib.sha1(rec["fl_problem"].encode()).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            records.append(rec)

    stats = {"inputs": total_in, "duplicates": duplicates, "processed": 0, "ok": 0,
             "rejected_placement": 0, "rejected_unsupported": 0, "rejected_verify": 0, "rejected_goal": 0, "error": 0}
    reasons, timing, errors = {}, {"convert": 0.0, "placement": 0.0, "verify": 0.0, "figures": 0.0}, []
    t_start = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=init, initargs=(str(out),)) as pool, \
            open(out / "problems.jsonl", "w") as f, open(out / "rejected.jsonl", "w") as rej:
        for status, payload, t in pool.imap_unordered(process, records, chunksize=4):
            stats["processed"] += 1
            stats[status] += 1
            for k, v in t.items():
                timing[k] += v
            if status == "ok":
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if limit and stats["ok"] >= limit:
                    pool.terminate()
                    break
            else:
                key = status
                if status == "rejected_verify":
                    key = "rejected_verify:" + payload["reason"]
                reasons[key] = reasons.get(key, 0) + 1
                rej.write(json.dumps({"status": status, **payload}, ensure_ascii=False) + "\n")
                if len(errors) < 5:
                    errors.append({"status": status, **{k: v for k, v in payload.items() if k != "record"}})
    wall = time.time() - t_start

    # 容量
    problems_bytes = (out / "problems.jsonl").stat().st_size
    figs_bytes = sum(p.stat().st_size for p in (out / "figs").rglob("*.png")) if (out / "figs").exists() else 0
    n_figs = sum(1 for _ in (out / "figs").rglob("*.png")) if (out / "figs").exists() else 0
    turns, expanded_bytes = 0, 0
    from vlma.steps import expand
    rows = [json.loads(l) for l in open(out / "problems.jsonl")]
    for c in rows:
        turns += c["n_turns"]
    for c in rows[:200]:  # 展開後のサイズは先頭200問で見積もる
        expanded_bytes += sum(len(json.dumps(s, ensure_ascii=False)) + 1 for s in expand(c))
    sample_n = min(len(rows), 200)
    seeds = {}
    for c in rows:
        seeds[c["placement_seed"]] = seeds.get(c["placement_seed"], 0) + 1
    stats.update({
        "wall_s": round(wall, 1),
        "problems_per_hour_wall": round(stats["ok"] / wall * 3600) if wall else None,
        "workers": workers,
        "cpu_seconds": {k: round(v, 1) for k, v in timing.items()},
        "turns_total": turns,
        "turns_per_problem": round(turns / max(len(rows), 1), 1),
        "problems_jsonl_bytes": problems_bytes,
        "figures": n_figs, "figures_bytes": figs_bytes,
        "figures_bytes_per_problem": round(figs_bytes / max(len(rows), 1)),
        "expanded_bytes_per_problem_est": round(expanded_bytes / max(sample_n, 1)),
        "placement_seed_histogram": dict(sorted(seeds.items())),
        "reject_reasons": reasons, "first_errors": errors,
    })
    json.dump(stats, open(out / "stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
