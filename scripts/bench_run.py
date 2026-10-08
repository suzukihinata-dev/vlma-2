"""ベンチマーク: モデルに問題を解かせ、環境（DDAR の検証器）で1手ずつ検証して、成功率などを測る。

usage (train コンテナ内。環境サーバー bench_env.sh が、別のコンテナで動いていること):
  python bench_run.py --model /artifacts/r3/run1/model-final --format turn --name run1 [--set bench_200.json] [--retries 0] [--batch 16]

エピソード: 環境が返す状態（問題・情報欄・過去の手・図）から、学習データと同じ形のプロンプトを作り、モデルが1手を出す。
環境が検証して、受理すれば状態が進み、却下すれば状態は変わらない。目標に届くか、試行の上限（参照証明の手数から決める）で終わる。
--retries R: 1問で却下してよい回数。既定の -1 は、回数では打ち切らない（却下も1ターンを使い、試行の上限＝ターン予算が尽きるまで続ける）。
  0 なら、最初の却下で、その問題は失敗（厳密）。却下された状態では、サンプリング（--temperature）で出し直す。受理された手は、必ず貪欲（temperature 0）で出す。
  乱数は --seed で固定する。
結果: /artifacts/bench/<name>/results.jsonl（1問1行。途中から再開できる）と、summary.json。
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

for p in ("/shared", str(Path(__file__).resolve().parents[2] / "_shared")):
    if Path(p).is_dir():
        sys.path.insert(0, p)

from vlma import bench  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--format", choices=list(bench.FORMATS), required=True, help="学習した方式: turn（A'）か seq（B）")
ap.add_argument("--name", required=True)
ap.add_argument("--data", default="/artifacts/genesisgeo/r2/build-100k")
ap.add_argument("--set", default="bench_200.json")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--env", default="http://vlma-env")
ap.add_argument("--ports", default="8100-8105")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--retries", type=int, default=-1, help="1問で却下してよい回数。-1（既定）: 回数では打ち切らず、試行の上限（ターン予算）だけで終える。0: 最初の却下で失敗（厳密）")
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--max-new", type=int, default=192)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-iters", type=int, default=0, help="試験用: この回数で打ち切る（時間の内訳を測るとき。結果は保存しない）")
ap.add_argument("--max-prompt", type=int, default=16384, help="これを超えるプロンプトになった問題は、失敗として打ち切る")
ap.add_argument("--slack-every", type=float, default=30, help="Slack に進捗を送る間隔（分）。0 なら送らない")
args = ap.parse_args()

out_dir = Path("/artifacts/bench") / args.name
out_dir.mkdir(parents=True, exist_ok=True)
lo, hi = (int(x) for x in args.ports.split("-"))
ports = list(range(lo, hi + 1))


def post(port, path, body, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(f"{args.env}:{port}{path}", data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=900).read())
        except Exception as e:  # noqa: BLE001
            body = e.read().decode("utf-8", "replace")[:400] if hasattr(e, "read") else ""  # 環境サーバーのエラーの本文
            if k == tries - 1:
                raise RuntimeError(f"env {path} failed: {type(e).__name__}: {e} {body}") from e
            time.sleep(2)


pids = json.load(open(Path(args.data) / args.set))
if args.limit:
    pids = pids[: args.limit]
res_path = out_dir / "results.jsonl"
done_pids = {json.loads(l)["problem_id"] for l in open(res_path)} if res_path.exists() else set()
queue = [p for p in pids if p not in done_pids]
print(f"problems {len(pids)}, already done {len(done_pids)}, to run {len(queue)}", flush=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

torch.manual_seed(args.seed)
processor = AutoProcessor.from_pretrained(args.model)
processor.tokenizer.padding_side = "left"
tok = processor.tokenizer
im_end, pad = tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id
model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
model.config.use_cache = True


prof = {"gen_s": 0.0, "env_s": 0.0, "new_tokens_max": 0, "prompt_tokens": 0, "iters": 0}  # 時間の内訳（生成・環境）と、プロンプトの長さ


@torch.no_grad()
def generate(prompts, images, sample):
    """prompts: 文章の一覧、images: 各文章の画像のパスの一覧。左詰めで、まとめて生成する。"""
    flat = [Image.open(p).convert("RGB") for ims in images for p in ims]
    enc = processor(text=prompts, images=flat or None, return_tensors="pt", padding=True).to("cuda")
    kw = dict(do_sample=True, temperature=args.temperature, top_p=0.95) if sample else dict(do_sample=False)
    t = time.time()
    out = model.generate(**enc, max_new_tokens=args.max_new, eos_token_id=im_end, pad_token_id=pad, **kw)
    torch.cuda.synchronize()
    new = out[:, enc["input_ids"].shape[1]:]
    prof["gen_s"] += time.time() - t
    prof["new_tokens_max"] = max(prof["new_tokens_max"], int(new.shape[1]))
    prof["prompt_tokens"] += int(enc["attention_mask"].sum())
    return [bench.clean_output(tok.decode(o, skip_special_tokens=False)) for o in new], int(enc["input_ids"].shape[1])


class Slot:
    def __init__(self, pid, i):
        self.pid, self.port = pid, ports[i % len(ports)]
        r = post(self.port, "/reset", {"problem_id": pid})
        self.eid, self.state, self.fmt = r["episode"], r["state"], bench.FORMATS[args.format]()
        self.rejections, self.resample, self.error = 0, False, None


def finish(slot, abandoned=False):
    try:
        r = post(slot.port, "/close", {"episode": slot.eid})["result"]
    except Exception as e:  # noqa: BLE001
        r = {"problem_id": slot.pid, "success": False, "kind": "error", "error": str(e)[:200]}
    r["abandoned"] = abandoned
    if slot.error:
        r["error"] = slot.error
    with open(res_path, "a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return r


slots, finished, n_next, t0, last_slack = [], [], 0, time.time(), time.time()
reporter = None
if args.slack_every:
    from slack_templates import Reporter  # noqa: E402

    reporter = Reporter("vlma-2", f"bench-{args.name}", thread_file=out_dir / "slack_thread.json")
    reporter.start({"モデル": args.model, "方式": args.format, "問題数": len(queue), "却下してよい回数": args.retries}, note="ベンチマーク（成功率: 目標に届いた問題の割合）")
pool = cf.ThreadPoolExecutor(max_workers=len(ports) * 2)
while queue or slots:
    if args.max_iters and prof["iters"] >= args.max_iters:
        print("max-iters reached; stopping (results not summarized)", flush=True)
        sys.exit(0)
    while queue and len(slots) < args.batch:
        slots.append(Slot(queue.pop(0), n_next))
        n_next += 1
    jobs = []  # (slot, 文章, 画像)
    for s in slots[:]:
        text, imgs = s.fmt.prompt(s.state)
        jobs.append((s, text, imgs))
    outputs = {}
    for sample in (False, True):
        grp = [j for j in jobs if j[0].resample == sample]
        if grp:
            outs, n_tok = generate([j[1] for j in grp], [j[2] for j in grp], sample)
            for (s, _, _), o in zip(grp, outs):
                outputs[id(s)] = o
            if n_tok > args.max_prompt:  # 長すぎるプロンプトは、メモリを使い切るので、この問題を打ち切る
                for s, _, _ in grp:
                    s.error = f"prompt too long ({n_tok} tokens)"
    t_env = time.time()
    futs = {pool.submit(post, s.port, "/step", {"episode": s.eid, "output": outputs[id(s)]}): (s, o) for s, _, _ in jobs
            for o in [outputs[id(s)]] if not s.error}
    for s, _, _ in jobs:
        if s.error:
            finished.append(finish(s, abandoned=True))
            slots.remove(s)
    for fut, (s, o) in futs.items():
        try:
            r = fut.result()
        except Exception as e:  # noqa: BLE001
            s.error = str(e)[:200]
            finished.append(finish(s, abandoned=True))
            slots.remove(s)
            continue
        if r["verdict"]["ok"]:
            s.fmt.accept(s.state, o)
            s.resample = False
        else:
            s.rejections += 1
            s.resample = True
        s.state = r["state"]
        if r["done"]:
            finished.append(finish(s))
            slots.remove(s)
        elif args.retries >= 0 and s.rejections > args.retries:
            finished.append(finish(s, abandoned=True))
            slots.remove(s)
    prof["env_s"] += time.time() - t_env
    prof["iters"] += 1
    if time.time() - last_slack > 60 * (args.slack_every or 1e9) and reporter:
        ok = sum(r.get("success", False) for r in finished)
        reporter.job(len(finished), len(finished) + len(queue) + len(slots), eta_h=None, extra={"成功": f"{ok}/{len(finished)}"})
        last_slack = time.time()
    print(f"done {len(finished)} active {len(slots)} queue {len(queue)} success {sum(r.get('success', False) for r in finished)} "
          f"elapsed {(time.time() - t0) / 60:.1f}min", flush=True)
    if prof["iters"] % 10 == 0:  # 生成と環境の、1回あたりの平均秒。最長の生成のトークン数。プロンプトの平均トークン数
        print(f"profile per iteration: gen {prof['gen_s'] / prof['iters']:.1f}s env {prof['env_s'] / prof['iters']:.1f}s "
              f"max_new_tokens_seen {prof['new_tokens_max']} prompt_tokens/iter {prof['prompt_tokens'] / prof['iters']:.0f}", flush=True)

all_results = [json.loads(l) for l in open(res_path)]
all_results = [r for r in all_results if r.get("kind") != "error" and r["problem_id"] in set(pids)]
errors = sum(json.loads(l).get("kind") == "error" for l in open(res_path))
summary = bench.summarize(all_results) | {"errors": errors, "model": args.model, "format": args.format, "retries": args.retries,
                                          "minutes": round((time.time() - t0) / 60, 1)}
json.dump(summary, open(out_dir / "summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False), flush=True)
if reporter:
    reporter.finish({"成功率": f"{summary['success_rate']:.1%}", "却下されなかった手の割合": f"{summary['move_accept_rate']:.1%}", "問題数": summary["problems"]},
                    outputs=[str(out_dir / "summary.json")], elapsed_min=summary["minutes"])
