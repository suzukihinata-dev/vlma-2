"""R2: 生成データから、指定した述語を結論に持つ手を含む、短い記録を1件選んで JSON で出す（テスト用データの取り出し）。

usage: python r2_pick_record.py <gen.jsonl> <pred> [--max-steps N]
"""
import json
import re
import sys

pred = sys.argv[2]
max_steps = int(sys.argv[sys.argv.index("--max-steps") + 1]) if "--max-steps" in sys.argv else 40
for line in open(sys.argv[1]):
    r = json.loads(line)
    out = r["llm_output_renamed"]
    proof = re.search(r"<proof>(.*?)</proof>", out, re.S).group(1)
    steps = [s.strip() for s in proof.split(";") if s.strip()]
    aux = re.search(r"<aux>(.*?)</aux>", out, re.S).group(1)
    segs = [s for s in aux.split(";") if s.strip()]
    if (len(steps) <= max_steps and len(segs) == 1 and any(s.startswith(pred + " ") for s in steps)
            and len(re.findall(r"\[\d+\]", segs[0])) <= 2):
        print(json.dumps({k: r[k] for k in ("fl_problem", "llm_input_renamed", "llm_output_renamed")}, ensure_ascii=False))
        break
