"""R2: Qwen3.5-0.8B のトークナイザーと画像処理の設定を取得し、学習サンプルのトークン数を測る（重みは取らない）。

usage: python r2_token_lengths.py <problems.jsonl> [--limit N] [--repo Qwen/Qwen3.5-0.8B-Base]
サンプルの並べ方（仮）: 画像 + [問題, 目標, 情報, 過去の操作列] → 出力。トークン数 = 文字のトークン数 + 画像のトークン数。
"""
import json
import statistics
import sys

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from vlma.steps import expand

repo = sys.argv[sys.argv.index("--repo") + 1] if "--repo" in sys.argv else "Qwen/Qwen3.5-0.8B-Base"
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 1500
path = snapshot_download(repo, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model", "merges.txt", "vocab.json"], local_dir="/artifacts/models-meta/" + repo.split("/")[-1])
print("downloaded to", path)
cfg = json.load(open(f"{path}/config.json"))
vis = cfg.get("vision_config", {})
print("model_type:", cfg.get("model_type"), "| architectures:", cfg.get("architectures"))
print("vision: patch", vis.get("patch_size"), "merge", vis.get("spatial_merge_size"))
try:
    pre = json.load(open(f"{path}/preprocessor_config.json"))
    print("preprocessor:", {k: pre.get(k) for k in ("patch_size", "merge_size", "min_pixels", "max_pixels", "size", "image_processor_type")})
except FileNotFoundError:
    pre = {}
    print("no preprocessor_config.json")
tok = AutoTokenizer.from_pretrained(path)
print("tokenizer:", type(tok).__name__, "vocab", len(tok))

ps = vis.get("patch_size", 16) or 16
merge = vis.get("spatial_merge_size", 2) or 2
image_tokens = (512 // ps) ** 2 // (merge * merge)  # 512x512 を縮小・拡大せずに処理した場合
print("image tokens for a 512x512 image (no resize):", image_tokens)

text_lens, outs, infos, hists, probs = [], [], [], [], []
for n, line in enumerate(open(sys.argv[1])):
    if n >= limit:
        break
    c = json.loads(line)
    steps = expand(c)
    # 文字のトークン数は、手ごとの出力と情報欄が累積的に増えるので、差分から全サンプルぶんを求める
    probs.append(len(tok(c["problem"] + " " + c["goal"], add_special_tokens=False)["input_ids"]))
    hist_tokens = 0
    for s in steps:
        o = len(tok(s["output"], add_special_tokens=False)["input_ids"])
        i = len(tok(s["info"], add_special_tokens=False)["input_ids"]) if s["info"] else 0
        text_lens.append(probs[-1] + i + hist_tokens + o)
        outs.append(o)
        infos.append(i)
        hists.append(hist_tokens)
        hist_tokens += o + 1
q = lambda a, p: sorted(a)[int(p * (len(a) - 1))]
total = [t + image_tokens for t in text_lens]
print(json.dumps({"problems": min(limit, n + 1), "samples": len(total),
                  "output_tokens": {"median": q(outs, .5), "p99": q(outs, .99), "max": max(outs)},
                  "info_tokens": {"median": q(infos, .5), "p90": q(infos, .9), "max": max(infos)},
                  "history_tokens": {"median": q(hists, .5), "p90": q(hists, .9), "max": max(hists)},
                  "problem_tokens": {"median": q(probs, .5), "max": max(probs)},
                  "total_tokens_with_image": {"median": q(total, .5), "p90": q(total, .9), "p99": q(total, .99), "max": max(total)},
                  "share_over": {str(L): round(sum(t > L for t in total) / len(total), 4) for L in (2048, 4096, 8192)}}))
