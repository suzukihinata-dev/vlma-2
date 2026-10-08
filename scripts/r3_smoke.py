"""R3: Qwen3.5-0.8B-Base で、実際のサンプルを使い、学習の速度・メモリの測定と、過学習の確認を行う。

usage (train コンテナ内):
  python r3_smoke.py speed [--batch 1] [--steps 20] [--max-len 8192] [--no-kernels] [--no-ckpt]
  python r3_smoke.py overfit [--n 16] [--steps 150] [--lr 1e-5]
  python r3_smoke.py check    損失をかける位置だけ計算する損失が、モデル本来の（全位置で計算する）損失と一致するか、メモリがどれだけ減るかを確かめる
速度の測定: 先頭の数問のサンプルを使い、前向き・後ろ向き・更新を行う。最初の3ステップは、コンパイルなどを含むので除く。
過学習の確認: 固定した n 個のサンプルを繰り返し学習し、損失が下がることと、各サンプルの出力が正しく再現されることを見る。
"""
import argparse
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["speed", "overfit", "check"])
ap.add_argument("--repo", default="Qwen/Qwen3.5-0.8B-Base")
ap.add_argument("--data", default="/artifacts/genesisgeo/r2/build-100k")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--k", type=int, default=4, help="1問あたりのターン数（速度の測定のための試験用。本番の学習は全ターンを使い、問題数で量を絞る）。0 なら全ターン")
ap.add_argument("--no-kernels", action="store_true", help="fla と causal-conv1d を使わず、PyTorch の代替処理にする（速度の比較用）")
ap.add_argument("--no-ckpt", action="store_true", help="勾配チェックポイントを使わない")
args = ap.parse_args()

if args.no_kernels:  # transformers を読み込む前に、部品を読めなくする
    sys.modules["fla"] = None
    sys.modules["causal_conv1d"] = None

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

from vlma.data import VlmaDataset  # noqa: E402

torch.manual_seed(0)
processor = AutoProcessor.from_pretrained(args.repo)
pad = processor.tokenizer.pad_token_id
ds = VlmaDataset(args.data, "train", processor, turns_per_problem=(args.k or None), seed=0, max_problems=300, max_length=args.max_len)
print("dataset samples:", len(ds), flush=True)

t0 = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(args.repo, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
model.config.use_cache = False
if not args.no_ckpt:
    model.gradient_checkpointing_enable()
model.train()
print(f"model loaded in {time.time() - t0:.0f}s | params {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B | kernels={'off' if args.no_kernels else 'on'} | ckpt={'off' if args.no_ckpt else 'on'}", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))


def collate(items):
    n = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), n), pad, dtype=torch.long)
    lab = torch.full((len(items), n), -100, dtype=torch.long)
    att = torch.zeros((len(items), n), dtype=torch.long)
    mm = torch.zeros((len(items), n), dtype=torch.long)
    for i, x in enumerate(items):
        m = len(x["input_ids"])
        ids[i, :m], lab[i, :m], att[i, :m] = x["input_ids"], x["labels"], 1
        if "mm_token_type_ids" in x:
            mm[i, :m] = x["mm_token_type_ids"]
    batch = {"input_ids": ids, "labels": lab, "attention_mask": att, "mm_token_type_ids": mm}
    if any("pixel_values" in x for x in items):
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in items if "pixel_values" in x])
        batch["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in items if "image_grid_thw" in x])
    return batch


def loss_on_labels(batch):
    """損失をかける位置（labels が -100 でない位置）だけ、出力の確率を計算する。

    語彙が約25万あるので、全位置で計算すると、メモリと時間が大きい。損失をかけるのは全体の約1.2%なので、
    その位置の隠れ状態だけを取り出して lm_head にかける。
    """
    labels = batch["labels"]
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    h = model.model(**inputs).last_hidden_state  # (B, T, hidden)
    h, target = h[:, :-1], labels[:, 1:]  # 次のトークンを予測する
    pick = target != -100
    logits = model.lm_head(h[pick]).float()
    return torch.nn.functional.cross_entropy(logits, target[pick])


def step(batch):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = loss_on_labels(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item(), int(batch["attention_mask"].sum()), int((batch["labels"] != -100).sum())


if args.mode == "check":
    model.eval()  # 確認なので、勾配チェックポイントや dropout の影響を避ける
    rows = []
    for idx in (0, len(ds) // 3, 2 * len(ds) // 3, len(ds) - 1):
        b = {k: v.to("cuda") for k, v in collate([ds[idx]]).items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            full = model(**b).loss.item()  # モデル本来の損失（全位置の確率を計算する）
        mem_full = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ours = loss_on_labels(b).item()
        mem_ours = torch.cuda.max_memory_allocated() / 2**30
        rows.append((int(b["attention_mask"].sum()), full, ours, mem_full, mem_ours))
        print(f"tokens {rows[-1][0]:5d} | loss full {full:.5f} ours {ours:.5f} (diff {abs(full - ours):.1e}) | peak mem GB full {mem_full:.1f} ours {mem_ours:.1f}", flush=True)
    print("RESULT max loss difference:", max(abs(r[1] - r[2]) for r in rows))
    sys.exit(0)

if args.mode == "speed":
    torch.cuda.reset_peak_memory_stats()
    order = list(range(len(ds)))
    import random

    random.Random(0).shuffle(order)
    tokens = sup = 0
    times = []
    all_times = []
    step_tokens = []
    lens = []
    for s in range(args.steps + 3):
        items = [ds[order[(s * args.batch + j) % len(order)]] for j in range(args.batch)]
        b = collate(items)
        torch.cuda.synchronize()
        t = time.time()
        loss, n_tok, n_sup = step(b)
        torch.cuda.synchronize()
        dt = time.time() - t
        all_times.append(round(dt, 2))
        if s >= 3:
            times.append(dt)
            step_tokens.append(n_tok)
            tokens += n_tok
            sup += n_sup
            lens.append(b["input_ids"].shape[1])
        if s < 3 or s % 5 == 0:
            print(f"step {s} loss {loss:.3f} tokens {n_tok} time {dt:.2f}s", flush=True)
    # 外れ値（再コンパイルなど）を除いた、定常の速度: 中央値の2倍を超えるステップを除く
    med = sorted(times)[len(times) // 2]
    steady = [(t, n) for t, n in zip(times, step_tokens) if t <= 2 * med]
    steady_tps = sum(n for _, n in steady) / sum(t for t, _ in steady)
    print(f"STEADY tokens/s (excluding steps slower than 2x median): {steady_tps:.0f} from {len(steady)}/{len(times)} steps")
    srt = sorted(times)
    print(f"step time (s): median {srt[len(srt) // 2]:.2f}, max {srt[-1]:.2f}, slowest 5: {srt[-5:]}")
    print(f"RESULT batch={args.batch} steps={args.steps} mean_len={sum(lens) / len(lens):.0f} "
          f"tokens/s={tokens / sum(times):.0f} supervised_tokens/s={sup / sum(times):.1f} s/step={sum(times) / len(times):.2f} "
          f"peak_mem_GB={torch.cuda.max_memory_allocated() / 2**30:.1f}")
else:
    items = [ds[i * (len(ds) // args.n)] for i in range(args.n)]
    batch = collate(items)
    n_tokens = int(batch["attention_mask"].sum())
    print(f"overfit on {args.n} fixed samples ({n_tokens} tokens)", flush=True)
    for s in range(args.steps):
        loss, _, _ = step(batch)
        if s % 10 == 0 or s == args.steps - 1:
            print(f"step {s} loss {loss:.4f}", flush=True)
    # 各サンプルの出力が、そのまま再現されるか（貪欲に生成して、正解と比べる）
    model.eval()
    model.config.use_cache = True
    ok = 0
    for x in items:
        ex_len = int((x["labels"] != -100).sum())
        prompt_len = len(x["input_ids"]) - ex_len
        b = collate([{**x, "input_ids": x["input_ids"][:prompt_len], "labels": x["labels"][:prompt_len], "attention_mask": x["attention_mask"][:prompt_len],
                      "mm_token_type_ids": x["mm_token_type_ids"][:prompt_len]}])
        b.pop("labels")
        b = {k: v.to("cuda") for k, v in b.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**b, max_new_tokens=ex_len + 4, do_sample=False)
        got = processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=False).split("<|im_end|>")[0]
        want = processor.tokenizer.decode(x["input_ids"][prompt_len:], skip_special_tokens=False).split("<|im_end|>")[0]
        ok += got.strip() == want.strip()
    print(f"RESULT overfit exact_match={ok}/{args.n}")
