"""R2 段階4: 2つの作成結果を統合し、問題数をそろえる。

usage: python r2_merge_datasets.py --base DIR --add DIR --limit N
DIR/problems.jsonl と DIR/figs/ を持つディレクトリを2つ受け取り、--add の問題を --base に足す（問題 id の重複は除く）。
--add の図は --base に移す（コピーしない）。合計が N を超える分は、末尾から問題と図を削除する。
--base の元の problems.jsonl は problems.base-before-merge.jsonl に残す。結果の内訳は merge_manifest.json に書く。
"""
import json
import os
import shutil
import sys
from pathlib import Path


def arg(name, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1])


def main():
    base, add, limit = Path(arg("--base")), Path(arg("--add")), arg("--limit", int)
    shutil.copy(base / "problems.jsonl", base / "problems.base-before-merge.jsonl")
    rows = [json.loads(l) for l in open(base / "problems.jsonl") if l.strip()]
    n_base = len(rows)
    ids = {r["problem_id"] for r in rows}
    added, dup = [], 0
    for line in open(add / "problems.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["problem_id"] in ids:
            dup += 1
            continue
        ids.add(r["problem_id"])
        added.append(r)
    kept_added = added[: max(0, limit - n_base)]
    dropped = added[len(kept_added):]
    for r in kept_added:
        src, dst = add / "figs" / r["problem_id"], base / "figs" / r["problem_id"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    final = rows + kept_added
    trimmed = []
    if len(final) > limit:  # base だけで N を超える場合
        trimmed = final[limit:]
        final = final[:limit]
        for r in trimmed:
            shutil.rmtree(base / "figs" / r["problem_id"], ignore_errors=True)
    with open(base / "problems.jsonl", "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"problems": len(final), "from_base": min(n_base, len(final)), "from_add": len(kept_added),
                "duplicates_between_inputs": dup, "add_not_used": len(dropped), "trimmed_from_base": len(trimmed),
                "turns_total": sum(r["n_turns"] for r in final),
                "figures_total": sum(r["n_figures"] for r in final)}
    json.dump(manifest, open(base / "merge_manifest.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
