"""R3: Qwen3.5-0.8B-Base を、方式 A'（--format turn）または方式 B（--format seq）のデータで学習する（全パラメータ、bf16 の混合精度、勾配チェックポイント）。

usage (train コンテナ内):
  python r3_train.py --run run1 --subset subset_1000.json [--epochs 1] [--lr 1e-5] [--accum 32]
  python r3_train.py --run run1 --resume       中断した学習を、最後のチェックポイントから再開する
出力: /artifacts/r3/<run>/ に、log.jsonl（学習・検証の記録）、ckpt-last/（再開用: モデルと最適化の状態）、
      model-step<N>/（途中経過のモデル）、config.json。
損失は、出力と終わりの記号だけにかけ、その位置だけ出力の確率を計算する。各手の損失の平均（トークン単位）。
検証: 検証の分割の問題から、決まった（問題, ターン）で、損失と、次のトークンの正解率を測る。
忘却の確認: 一般文章（学習に使わない分）の perplexity を、学習前と途中・最後に測る。
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--repo", default="Qwen/Qwen3.5-0.8B-Base")
ap.add_argument("--data", default="/artifacts/genesisgeo/r2/build-100k")
ap.add_argument("--subset", default="subset_1000.json")
ap.add_argument("--format", choices=["turn", "seq"], default="turn", help="turn: 方式 A'（1ターン = 1サンプル）。seq: 方式 B（1問 = 1本の系列、全ターンの出力に損失）")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--min-lr-ratio", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=50)
ap.add_argument("--accum", type=int, default=32, help="1回の更新に使うサンプル数（1サンプルずつ計算して積み上げる）")
ap.add_argument("--max-len", type=int, default=12288, help="これを超えるサンプルは飛ばす")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--val-every", type=int, default=250, help="検証の間隔（更新の回数）")
ap.add_argument("--save-every", type=int, default=250)
ap.add_argument("--val-problems", type=int, default=100)
ap.add_argument("--val-turns", type=int, default=6)
ap.add_argument("--general-docs", type=int, default=200, help="忘却の確認に使う一般文章の数（0 なら測らない）")
ap.add_argument("--max-steps", type=int, default=0, help="試験用: この更新回数で打ち切る")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

import torch  # noqa: E402
import torch.utils.checkpoint  # noqa: E402,F401
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

from vlma.data import VlmaDataset, VlmaSeqDataset  # noqa: E402

out = Path("/artifacts/r3") / args.run
out.mkdir(parents=True, exist_ok=True)
random.seed(args.seed)
torch.manual_seed(args.seed)
processor = AutoProcessor.from_pretrained(args.repo)
pad = processor.tokenizer.pad_token_id


def log(**kw):
    kw["time"] = round(time.time(), 1)
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")
    print(json.dumps(kw, ensure_ascii=False), flush=True)


# ---- データ ----
Data = VlmaSeqDataset if args.format == "seq" else VlmaDataset  # seq の検証は、検証の問題の全ターン（--val-turns は使わない）
train_ds = Data(args.data, "train", processor, turns_per_problem=None, seed=args.seed, subset=Path(args.data) / args.subset,
                max_length=args.max_len)
val_ds = Data(args.data, "val", processor, turns_per_problem=args.val_turns, seed=1234, max_problems=args.val_problems,
              max_length=args.max_len)
n_samples = len(train_ds)
steps_per_epoch = n_samples // args.accum
total_steps = steps_per_epoch * args.epochs
if args.max_steps:
    total_steps = min(total_steps, args.max_steps)
print(f"train problems {len(train_ds.problems)} samples/epoch {n_samples} -> {steps_per_epoch} updates/epoch, total {total_steps}", flush=True)


class OrderedView(torch.utils.data.Dataset):
    """epoch ごとの、全サンプルの順番（混ぜてある）。start 番目から先を使う。長すぎるサンプルは None を返す。"""

    def __init__(self, ds, epoch, start):
        order = list(range(len(ds)))
        random.Random(f"{args.seed}:{epoch}").shuffle(order)
        self.ds, self.order, self.start = ds, order, start

    def __len__(self):
        return len(self.order) - self.start

    def __getitem__(self, k):
        try:
            return self.ds[self.order[self.start + k]]
        except ValueError:  # 長すぎる
            return None


class Loader:
    """DataLoader（複数のプロセスで、展開・画像・トークン化を先読み）から、1サンプルずつ取り出す。位置は再開のために数える。"""

    def __init__(self, ds, epoch, start):
        self.pos, self.skipped = start, 0
        view = OrderedView(ds, epoch, start)
        self._it = iter(torch.utils.data.DataLoader(view, batch_size=None, shuffle=False, num_workers=args.workers,
                                                    prefetch_factor=8, persistent_workers=False))

    def __next__(self):
        while True:
            item = next(self._it)  # 終わりなら StopIteration
            self.pos += 1
            if item is None:
                self.skipped += 1
                continue
            return item


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
    b = {"input_ids": ids, "labels": lab, "attention_mask": att, "mm_token_type_ids": mm}
    if any("pixel_values" in x for x in items):
        b["pixel_values"] = torch.cat([x["pixel_values"] for x in items if "pixel_values" in x])
        b["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in items if "image_grid_thw" in x])
    return b


# ---- モデル ----
model = Qwen3_5ForConditionalGeneration.from_pretrained(args.repo, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.train()
decay, no_decay = [], []
for n_, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.0}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.95), fused=True)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, total_steps - args.warmup)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


LOSS_CHUNK = 1024  # 損失をかける位置が多い（方式 B: 1系列に約2,000）と、語彙（約25万）ぶんの logits がメモリを埋めるので、この数ずつ計算する


def _chunk_nll(hc, tc):
    logits = model.lm_head(hc).float()
    return torch.nn.functional.cross_entropy(logits, tc, reduction="sum"), (logits.argmax(-1) == tc).sum()


def loss_and_acc(batch, train=True):
    """損失をかける位置だけ、出力の確率を計算する。損失（トークン平均）と、その位置の正解数・個数を返す。
    位置が多いときは、LOSS_CHUNK 個ずつ、チェックポイント付きで計算する（logits を保持しない。結果は変わらない）。"""
    labels = batch["labels"]
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    h = model.model(**inputs).last_hidden_state
    h, target = h[:, :-1], labels[:, 1:]
    pick = target != -100
    hs, ts = h[pick], target[pick]
    nll, correct = 0.0, 0
    for a in range(0, len(ts), LOSS_CHUNK):
        hc, tc = hs[a:a + LOSS_CHUNK], ts[a:a + LOSS_CHUNK]
        if torch.is_grad_enabled():
            c_nll, c_ok = torch.utils.checkpoint.checkpoint(_chunk_nll, hc, tc, use_reentrant=False)
        else:
            c_nll, c_ok = _chunk_nll(hc, tc)
        nll, correct = nll + c_nll, correct + int(c_ok)
    return nll / len(ts), correct, len(ts)


# ---- 検証と、忘却の確認 ----
@torch.no_grad()
def validate():
    model.eval()
    tot_loss = tot_tok = correct = 0
    for i in range(len(val_ds)):
        try:
            b = {k: v.to("cuda") for k, v in collate([val_ds[i]]).items()}
        except ValueError:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ok, n = loss_and_acc(b, train=False)
        tot_loss += loss.item() * n
        tot_tok += n
        correct += ok
    model.train()
    return {"val_loss": tot_loss / tot_tok, "val_token_acc": correct / tot_tok, "val_tokens": tot_tok}


general = []
if args.general_docs:
    from vlma.general import load_general_text

    general = load_general_text(args.general_docs, processor.tokenizer)


@torch.no_grad()
def general_ppl():
    """一般文章（学習に使わない）の perplexity。全位置に損失をかける。"""
    if not general:
        return {}
    model.eval()
    nll = n = 0
    for ids in general:
        x = ids.to("cuda")[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[:, :-1]
            logits = model.lm_head(h).float()
        nll += torch.nn.functional.cross_entropy(logits[0], x[0, 1:], reduction="sum").item()
        n += x.shape[1] - 1
    model.train()
    return {"general_ppl": math.exp(nll / n), "general_tokens": n}


def save(path, with_opt):
    path.mkdir(parents=True, exist_ok=True)
    if with_opt:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch, "pos": loader.pos}, path / "state.pt")
    else:
        model.to(torch.bfloat16).save_pretrained(path)
        processor.save_pretrained(path)
        model.to(torch.float32)


# ---- 学習 ----
step, epoch, start_pos = 0, 0, 0
if args.resume and (out / "ckpt-last" / "state.pt").exists():
    st = torch.load(out / "ckpt-last" / "state.pt", map_location="cuda")
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    step, epoch, start_pos = st["step"], st["epoch"], st["pos"]
    print(f"resumed at step {step} epoch {epoch} pos {start_pos}", flush=True)
json.dump(vars(args) | {"total_steps": total_steps, "samples_per_epoch": n_samples}, open(out / "config.json", "w"), indent=1)
if step == 0:
    log(event="baseline", step=0, **validate(), **general_ppl())

loader = Loader(train_ds, epoch, start_pos)
first_step = step
t_start, tok_acc, loss_acc, n_acc = time.time(), 0, 0.0, 0
while step < total_steps:
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    tok = 0
    sup_loss, sup_tok = 0.0, 0
    batch_items = []
    try:
        for _ in range(args.accum):
            batch_items.append(next(loader))
    except StopIteration:
        epoch += 1
        if epoch >= args.epochs:
            break
        loader = Loader(train_ds, epoch, 0)
        continue
    # 更新1回ぶんの損失は、含まれる全サンプルの、損失をかけるトークンの平均（トークン数で重み付け）
    n_sup_total = sum(int((x["labels"] != -100).sum()) for x in batch_items)
    for x in batch_items:
        b = {k: v.to("cuda") for k, v in collate([x]).items()}
        n_sup = int((b["labels"] != -100).sum())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, ok, n = loss_and_acc(b)
        (loss * n_sup / n_sup_total).backward()
        sup_loss += loss.item() * n_sup
        sup_tok += n_sup
        tok += int(b["attention_mask"].sum())
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    tok_acc += tok
    loss_acc += sup_loss
    n_acc += sup_tok
    if step % 10 == 0 or step == total_steps:
        el = time.time() - t_start
        el = time.time() - t_start
        done_steps = step - first_step
        log(event="train", step=step, epoch=epoch, loss=round(loss_acc / n_acc, 4), grad_norm=round(gn, 3), lr=round(lr_at(step), 8),
            tokens_per_s=round(tok_acc / el), eta_h=round((total_steps - step) * el / max(1, done_steps) / 3600, 2),
            skipped_long=loader.skipped)
        loss_acc, n_acc = 0.0, 0
    if step % args.val_every == 0 or step == total_steps:
        log(event="val", step=step, **validate(), **general_ppl())
    if step % args.save_every == 0 or step == total_steps:
        save(out / "ckpt-last", with_opt=True)
        if step == total_steps or step % (args.save_every * 4) == 0:
            save(out / f"model-step{step}", with_opt=False)
save(out / "model-final", with_opt=False)
log(event="done", step=step, minutes=round((time.time() - t_start) / 60, 1))
