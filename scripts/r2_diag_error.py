"""R2 段階4: エラーになった問題を1件ずつ処理して、完全なトレースバックを出す。

usage: python r2_diag_error.py <in.jsonl>
"""
import json
import sys
import traceback

from vlma.figures import find_placement_seed
from vlma.steps import to_steps
from vlma.verify import StepVerifier

for line in open(sys.argv[1]):
    rec = json.loads(line)
    try:
        seed = find_placement_seed(rec)
        print("seed", seed)
        v = StepVerifier(rec, seed=seed or 0)
        for s in to_steps(rec):
            verdict = v.check(s["output"])
            if not verdict.ok:
                print("rejected", s["output"][:80], verdict)
                break
    except Exception:  # noqa: BLE001
        print(traceback.format_exc()[-1500:])
