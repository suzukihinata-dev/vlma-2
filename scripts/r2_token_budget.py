"""R2: 学習データの作り方による、トークン数の総量の違いを測る。

usage: python r2_token_budget.py <problems.jsonl> [--limit N]
  A: 1手ごとに独立したサンプル（毎回、図・問題・情報・過去の操作列を入れ直す。いまの設計）
  A2: Aと同じ独立したサンプルだが、画像は「最初のターンと、作図の手のあと」だけ入れる（図が変わったターン）
  B: 1問を1本の会話にする（図は作図で変わったときだけ入れる。損失は全ての手の出力にかける）
画像は 256 トークン、手ごとの区切りは仮に8トークンとする。
"""
import json
import sys

from transformers import AutoTokenizer

from vlma.steps import expand

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 500
tok = AutoTokenizer.from_pretrained("/artifacts/models-meta/Qwen3.5-0.8B-Base")
n = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0
IMG, MARK = 256, 8
a_total = a2_total = b_total = samples = turns = changes = img_turns = 0
for i, line in enumerate(open(sys.argv[1])):
    if i >= limit:
        break
    c = json.loads(line)
    steps = expand(c)
    prob = n(c["problem"] + " " + c["goal"])
    hist = 0
    for k, s in enumerate(steps):
        o = n(s["output"])
        a_total += IMG + prob + n(s["info"]) + hist + o + MARK
        show_img = k == 0 or steps[k - 1]["output"].startswith("CONSTRUCT")  # 図が変わったターン
        img_turns += show_img
        a2_total += (IMG if show_img else 0) + prob + n(s["info"]) + hist + o + MARK
        hist += o + 1
        samples += 1
    turns += len(steps)
    b_total += prob + sum(n(s["output"]) + MARK for s in steps) + n(steps[-1]["info"]) + IMG * c["n_figures"]
    changes += c["n_figures"] - 1
print(json.dumps({
    "problems": min(limit, i + 1), "turns": turns, "image_changes": changes,
    "image_change_share_of_turns": round(changes / turns, 4),
    "A_tokens_per_problem": round(a_total / min(limit, i + 1)),
    "B_tokens_per_problem": round(b_total / min(limit, i + 1)),
    "A_over_B": round(a_total / b_total, 1),
    "A2_tokens_per_problem": round(a2_total / min(limit, i + 1)),
    "A2_saving_vs_A": round(1 - a2_total / a_total, 3),
    "A2_image_turn_share": round(img_turns / turns, 4),
    "A2_tokens_for_100k_problems_billion": round(a2_total / min(limit, i + 1) * 100000 / 1e9, 1),
    "images_per_problem_A_vs_A2": [round(turns / min(limit, i + 1), 1), round(img_turns / min(limit, i + 1), 1)],
    "A_tokens_for_100k_problems_billion": round(a_total / min(limit, i + 1) * 100000 / 1e9, 1),
    "B_tokens_for_100k_problems_billion": round(b_total / min(limit, i + 1) * 100000 / 1e9, 2),
}))
