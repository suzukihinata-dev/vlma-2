"""R3: 学習の記録（log.jsonl）を見て、Slack に進捗を知らせる。学習とは別のプロセスで動かす。

usage: python r3_watch.py --run run1 [--interval 30] [--stale 25] [--once]
  定期的に（--interval 分ごと）、進み具合・損失・速度・残り時間を送り、新しい検証があれば、その結果（学習前との変化つき）と、進捗のグラフ（PNG）を送る。終了時にも、グラフを送る。
  記録が --stale 分以上更新されなければ、止まっている可能性を知らせる。学習が終われば、完了の報告をして終わる。
  メッセージの形は、research/_shared/slack_templates.py のテンプレート（全プロジェクト共通）。
  Slack の設定（vlma/notify.py を参照）が無ければ、標準出力に出すだけ。--once: 1回だけ送って終わる（試験用）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

for p in ("/shared", str(Path(__file__).resolve().parents[2] / "_shared")):  # コンテナでは /shared、手元では research/_shared
    if Path(p).is_dir():
        sys.path.insert(0, p)
from slack_templates import Reporter  # noqa: E402

from vlma import notify  # noqa: E402

PROJECT = "vlma-2"
LOWER_IS_BETTER = ("検証の損失", "一般文章の perplexity")
WARN_IF_WORSE = {"一般文章の perplexity": 10}  # 学習前から10%を超えて悪化したら警告（ブレンドの検討）


def read_log(path: Path):
    ev = []
    if path.exists():
        for line in open(path):
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


def last(ev, name):
    xs = [e for e in ev if e.get("event") == name]
    return xs[-1] if xs else None


def metrics_of(e):
    m = {"検証の損失": e["val_loss"], "検証の正解率": e["val_token_acc"]}
    if e.get("general_ppl"):
        m["一般文章の perplexity"] = e["general_ppl"]
    return m


def panels_of(ev):
    """log.jsonl から、グラフのパネル（_shared/slack_charts.py の形）を作る。グラフ内の文字は英語。"""
    tr = [e for e in ev if e.get("event") == "train"]
    va = [e for e in ev if e.get("event") == "val"]
    base = last(ev, "baseline")
    bs = [base] + va if base else va  # 学習前を、step 0 の点として先頭に足す
    ser = lambda rows, key: {"x": [e["step"] for e in rows if e.get(key)], "y": [e[key] for e in rows if e.get(key)]}
    hl = lambda key, label: [{"y": base[key], "label": label}] if base and base.get(key) else []
    return [
        {"title": "Training loss", "series": [dict(ser(tr, "loss"), label="train")], "logy": True},
        {"title": "Validation loss", "series": [dict(ser(bs, "val_loss"), label="val", marker=True)], "hlines": hl("val_loss", "before training")},
        {"title": "Validation token accuracy", "series": [dict(ser(bs, "val_token_acc"), label="val", marker=True)], "percent": True,
         "hlines": hl("val_token_acc", "before training")},
        {"title": "General text perplexity (forgetting)", "series": [dict(ser(bs, "general_ppl"), label="wikitext", marker=True)],
         "hlines": hl("general_ppl", "before training")},
        {"title": "Throughput (tokens/s)", "series": [dict(ser(tr, "tokens_per_s"), label="train")]},
        {"title": "Learning rate", "series": [dict(ser(tr, "lr"), label="lr")]},
    ]


def send_chart(r, ev, out, tag):
    """グラフを作って添付する。失敗しても学習の監視は止めない。"""
    try:
        r.chart(panels_of(ev), out / f"chart-{tag}.png")
    except Exception as e:  # noqa: BLE001
        print("[r3_watch] chart failed:", type(e).__name__, e, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--stale", type=float, default=25)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    out = Path("/artifacts/r3") / a.run
    thread = out / "slack_thread.json"  # 最初の通知がスレッドの親になり、以後はその返信に積む（Bot トークンのとき）
    r = Reporter(PROJECT, a.run, thread_file=thread, send=notify.send)
    fmts = {"検証の正解率": "{:.1%}"}
    if not thread.exists():
        cfg = {}
        try:
            c = json.load(open(out / "config.json"))
            cfg = {"学習率": c["lr"], "更新あたりのサンプル数": c["accum"], "エポック数": c["epochs"], "問題の一覧": c["subset"]}
            r.start(cfg, total=c["total_steps"], note=f"データ: 1エポック {c['samples_per_epoch']:,} サンプル。進捗は、{a.interval:.0f} 分ごとに、このスレッドに積みます。")
        except Exception:  # noqa: BLE001  config がまだ無いとき
            r.start(None, note=f"進捗は、{a.interval:.0f} 分ごとに、このスレッドに積みます。")
    reported_step, reported_val, last_alert = -1, -1, 0.0
    while True:
        ev = read_log(out / "log.jsonl")
        total = 0
        try:
            total = json.load(open(out / "config.json"))["total_steps"]
        except Exception:  # noqa: BLE001
            pass
        tr, base, va, done = last(ev, "train"), last(ev, "baseline"), last(ev, "val"), last(ev, "done")
        if tr and tr["step"] != reported_step:
            r.progress(tr["step"], total, loss=tr["loss"], speed=tr["tokens_per_s"], eta_h=tr["eta_h"])
            reported_step = tr["step"]
        if va and va["step"] != reported_val:
            r.evaluation(va["step"], metrics_of(va), baseline=metrics_of(base) if base else None, lower_is_better=LOWER_IS_BETTER,
                         warn_if_worse_pct=WARN_IF_WORSE, formats=fmts)
            reported_val = va["step"]
            send_chart(r, ev, out, f"step{va['step']}")
        if not tr and not va:
            notify.send(f"🚀 【{PROJECT} / {a.run}】まだ学習の記録がありません（起動中、または学習前の評価中）", thread_file=thread)
        if done:
            summary = {"更新": f"{done['step']:,}"}
            if va:
                summary.update({k: v for k, v in metrics_of(va).items()})
            if not va or va["step"] != done["step"]:  # 最後の検証と同じ step なら、直前に同じグラフを送っている
                send_chart(r, ev, out, "final")
            r.finish(summary, outputs=[str(out / "model-final")], elapsed_min=done["minutes"])
            return
        if a.once:
            return
        if ev:
            idle = (time.time() - ev[-1]["time"]) / 60
            if idle > a.stale and time.time() - last_alert > 3600:
                r.stalled(idle)
                last_alert = time.time()
        time.sleep(a.interval * 60)


if __name__ == "__main__":
    sys.exit(main())
