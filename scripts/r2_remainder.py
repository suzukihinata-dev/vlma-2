"""生成データのうち、problems.jsonl に入らなかった問題だけを取り出す。

usage: python r2_remainder.py <gen.jsonl> <problems.jsonl> <out.jsonl>
"""
import hashlib
import json
import sys

done = {json.loads(l)["problem_id"] for l in open(sys.argv[2]) if l.strip()}
n = 0
with open(sys.argv[3], "w") as f:
    for line in open(sys.argv[1]):
        if not line.strip():
            continue
        rec = json.loads(line)
        if hashlib.sha1(rec["fl_problem"].encode()).hexdigest()[:12] not in done:
            f.write(line)
            n += 1
print(n)
