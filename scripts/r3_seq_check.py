"""R3: 方式 B（1問 = 1本の系列）の確認と、計算量の見積もり。

usage (train コンテナ内): python r3_seq_check.py [--subset subset_1000.json] [--n 1000]
 - 各問題で、損失をかける位置が、全ターンの出力と一致するか（ターン数と同じ数の区間）を確認する。
 - 最初のターンの損失の位置が、方式 A' の1ターン目と同じトークンか確認する。
 - 1問あたりのトークン数（画像を含む）と、損失をかけるトークン数を数える。
"""
import json
import statistics as st
import sys
from pathlib import Path

from transformers import AutoProcessor

from vlma.data import VlmaDataset, VlmaSeqDataset


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


d = Path(arg("--data", "/artifacts/genesisgeo/r2/build-100k"))
subset = d / arg("--subset", "subset_1000.json")
n = arg("--n", 1000, int)
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
seq = VlmaSeqDataset(d, "train", proc, subset=subset, max_length=10**9)
prime = VlmaDataset(d, "train", proc, subset=subset, max_length=10**9)
first_turn = {}
for k, (i, t) in enumerate(prime.index):
    if t == 0:
        first_turn[prime.problems[i][0]] = k
tok, sup, imgs, turns, bad = [], [], [], [], 0
for j in range(min(n, len(seq))):
    item = seq[j]
    lab = item["labels"]
    tok.append(len(item["input_ids"]))
    sup.append(int((lab != -100).sum()))
    imgs.append(int(item["image_grid_thw"].shape[0]) if "image_grid_thw" in item else 0)
    turns.append(seq.problems[j][2])
    if j < 50:  # A' の1ターン目と、先頭の区間が同じか
        a = prime[first_turn[seq.problems[j][0]]]
        a_t = a["input_ids"][a["labels"] != -100].tolist()
        b_t = item["input_ids"][lab != -100].tolist()[: len(a_t)]
        bad += a_t != b_t
q = lambda a, p: sorted(a)[int(p * (len(a) - 1))]
res = {"problems": len(tok), "tokens_per_problem": {"mean": round(st.mean(tok)), "median": q(tok, .5), "p90": q(tok, .9), "max": max(tok)},
       "supervised_per_problem": {"mean": round(st.mean(sup))}, "supervised_share": round(sum(sup) / sum(tok), 3),
       "images_per_problem_mean": round(st.mean(imgs), 2), "turns_per_problem_mean": round(st.mean(turns), 1),
       "first_turn_mismatch_vs_A'(of 50)": bad, "over_8192": sum(x > 8192 for x in tok), "over_12288": sum(x > 12288 for x in tok)}
print(json.dumps(res, ensure_ascii=False))
