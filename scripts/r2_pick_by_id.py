"""R2: 問題 id（fl_problem の SHA-1 の先頭12文字）から、生成データの元の記録を取り出して JSON で出す（テスト用データ）。

usage: python r2_pick_by_id.py <gen.jsonl> <problem_id>
"""
import hashlib
import json
import sys

for line in open(sys.argv[1]):
    r = json.loads(line)
    if hashlib.sha1(r["fl_problem"].encode()).hexdigest()[:12] == sys.argv[2]:
        print(json.dumps({k: r[k] for k in ("fl_problem", "llm_input_renamed", "llm_output_renamed")}, ensure_ascii=False))
        break
