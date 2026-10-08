"""R3: 訓練/検証の分割のための、データの似た問題の分布を調べる。

usage: python r2_split_analysis.py <problems.jsonl>
似た問題の群（family）を、次のキーで数える。
  skeleton = 前提の作図の並び（座標を除く）+ 補助作図   … 同じ図の作り方
  skeleton + goal                                          … 同じ図の作り方で、同じ目標
"""
import collections
import hashlib
import json
import re
import sys

COORD = re.compile(r"@[-0-9.eE]+_[-0-9.eE]+")


def key_skeleton(c):
    prem = COORD.sub("", c["fl_problem"].split("?", 1)[0])
    aux = [a for a in c["actions"] if a.startswith("CONSTRUCT")]
    return hashlib.sha1((prem + "|" + "|".join(aux)).encode()).hexdigest()[:12]


def key_premises_only(c):
    return hashlib.sha1(COORD.sub("", c["fl_problem"].split("?", 1)[0]).encode()).hexdigest()[:12]


sk, sk_goal, prem = collections.Counter(), collections.Counter(), collections.Counter()
n = 0
for line in open(sys.argv[1]):
    c = json.loads(line)
    k = key_skeleton(c)
    sk[k] += 1
    sk_goal[k + c["goal"]] += 1
    prem[key_premises_only(c)] += 1
    n += 1
def summarize(cnt, name):
    sizes = sorted(cnt.values(), reverse=True)
    in_big = sum(s for s in sizes if s >= 2)
    print(f"{name}: 群の数 {len(cnt)}（問題 {n}）、最大の群 {sizes[0]}、2問以上の群に入る問題 {in_big} ({in_big / n:.1%})、"
          f"単独の問題 {sum(1 for s in sizes if s == 1)}")
summarize(prem, "前提の作図の並び（座標を除く）")
summarize(sk, "前提 + 補助作図")
summarize(sk_goal, "前提 + 補助作図 + 目標")
