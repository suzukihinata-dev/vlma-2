"""R2: 述語を演算子で書いたときのトークン数を、GenesisGeo-2B のトークナイザー（Qwen 系）で測る。

usage: python r2_notation_tokens.py <problems.jsonl> [--limit N]
手の出力（規則名つき）で、述語名をそのまま書く現在の形と、演算子で書く形のトークン数を比べる。
"""
import json
import re
import statistics
import sys

from transformers import AutoTokenizer

from vlma.actions import parse_action
from vlma.steps import expand

tok = AutoTokenizer.from_pretrained("/models/genesisgeo-2b")


def n(s):
    return len(tok.encode(s, add_special_tokens=False))


print("記号のトークン数:", {s: n(s) for s in ["//", "∥", "⊥", "┴", "_|_", "≅", "∽", "~", "∠", "△", "=", "perp", "para", "cong", "eqangle", "eqratio"]})
print("点の並び:", {s: n(s) for s in ["a b c", "abc", "A B C", "ABC"]})

SIMPLE = ("para", "perp", "cong")  # 括弧やカンマが要らない、4点の関係だけ演算子にする場合
VARIANTS = {
    "ascii": {"para": "//", "perp": "_|_", "cong": "=", "tri": "tri", "simtri": "~", "simtrir": "~r", "contri": "==", "contrir": "==r", "ang": "ang"},
    "unicode": {"para": "∥", "perp": "⊥", "cong": "=", "tri": "△", "simtri": "∽", "simtrir": "∽r", "contri": "≅", "contrir": "≅r", "ang": "∠"},
    "simple_ascii": {"para": "//", "perp": "_|_", "cong": "="},
    "simple_unicode": {"para": "∥", "perp": "⊥", "cong": "="},
    "simple_perp_box": {"para": "//", "perp": "┴", "cong": "="},
    "unicode_perp_box": {"para": "∥", "perp": "┴", "cong": "=", "tri": "△", "simtri": "∽", "simtrir": "∽r", "contri": "≅", "contrir": "≅r", "ang": "∠"},
}


def notate(pred, args, v):
    s = VARIANTS[v]
    a = list(args)
    if v.startswith("simple") and pred not in SIMPLE:
        return None
    if pred in ("para", "perp") and len(a) == 4:
        return f"{a[0]} {a[1]} {s[pred]} {a[2]} {a[3]}"
    if pred == "cong" and len(a) == 4:
        return f"{a[0]} {a[1]} = {a[2]} {a[3]}"
    if pred == "eqangle" and len(a) == 8:
        return f"{s['ang']}({a[0]} {a[1]}, {a[2]} {a[3]}) = {s['ang']}({a[4]} {a[5]}, {a[6]} {a[7]})"
    if pred == "eqratio" and len(a) == 8:
        return f"{a[0]} {a[1]} / {a[2]} {a[3]} = {a[4]} {a[5]} / {a[6]} {a[7]}"
    if pred in ("simtri", "simtrir", "contri", "contrir") and len(a) == 6:
        return f"{s['tri']} {a[0]} {a[1]} {a[2]} {s[pred]} {s['tri']} {a[3]} {a[4]} {a[5]}"
    if pred == "midp" and len(a) == 3:
        return f"{a[0]} = mid {a[1]} {a[2]}"
    return None  # 演算子にしない述語は、そのまま


limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 300
tot = {"current": [], **{v: [] for v in VARIANTS}}
covered = total = 0
for i, line in enumerate(open(sys.argv[1])):
    if i >= limit:
        break
    c = json.loads(line)
    for s in expand(c):
        o = s["output"]
        if not o.startswith("APPLY"):
            continue
        a = parse_action(o)
        total += 1
        tot["current"].append(n(o))
        for v in VARIANTS:
            body = notate(a.pred, a.args, v)
            if v == "ascii":
                covered += body is not None
            stmt = body if body is not None else f"{a.pred} {' '.join(a.args)}"
            prem = (" from " + " ".join(f"[{p}]" for p in a.premises)) if a.premises else ""
            tot[v].append(n(f"APPLY {a.rule} : {stmt} [{a.fact_id}]{prem}"))
print("手の数:", total, "演算子にした結論の割合: %.1f%%" % (100 * covered / total))
for k, v in tot.items():
    print(f"{k:18} 1手あたりの平均トークン数 {statistics.mean(v):.1f}")
