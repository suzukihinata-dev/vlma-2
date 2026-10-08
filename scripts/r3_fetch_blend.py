"""R3: ブレンド用の一般データを、必要な分だけ取得する（公開データ。再配布しない）。

usage: python r3_fetch_blend.py --out /artifacts/blend [--text-docs 20000] [--math-docs 8000]
  一般的な文章: HuggingFaceFW/fineweb-edu（odc-by）の sample/10BT の先頭から
  数学の文章:   open-web-math/open-web-math（odc-by）の先頭の shard から
  画像つき文章: yhshin1020/coco-img-caption-pairs（COCO の画像と説明。ライセンスの記載なし。COCO の画像は Flickr 由来）
parquet は、行グループごとに必要な分だけ読む。各データの取得量と、トークンの概算を、OUT/blend_summary.json に記録する。
"""
import gzip
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download


def arg(name, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def stream_docs(fs, path, columns, limit):
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        n = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=columns).to_pylist()
            for row in t:
                yield row
                n += 1
                if n >= limit:
                    return


def main():
    out = Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    summary = {}

    n_text = arg("--text-docs", 20000, int)
    docs = chars = toks = 0
    with gzip.open(out / "fineweb_edu.jsonl.gz", "wt") as g:
        for row in stream_docs(fs, "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet", ["id", "text", "token_count"], n_text):
            g.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")
            docs += 1
            chars += len(row["text"])
            toks += row["token_count"] or 0
    summary["fineweb_edu"] = {"docs": docs, "chars": chars, "tokens_reported": toks, "license": "odc-by", "source": "HuggingFaceFW/fineweb-edu sample/10BT/000_00000.parquet"}

    n_math = arg("--math-docs", 8000, int)
    files = sorted(p for p in fs.ls("datasets/open-web-math/open-web-math/data", detail=False) if p.endswith(".parquet"))
    docs = chars = 0
    with gzip.open(out / "openwebmath.jsonl.gz", "wt") as g:
        for row in stream_docs(fs, files[0], ["url", "text"], n_math):
            g.write(json.dumps({"url": row["url"], "text": row["text"]}, ensure_ascii=False) + "\n")
            docs += 1
            chars += len(row["text"])
    summary["openwebmath"] = {"docs": docs, "chars": chars, "license": "odc-by", "source": files[0]}

    path = snapshot_download("yhshin1020/coco-img-caption-pairs", repo_type="dataset", local_dir=str(out / "coco_pairs"))
    rows = sum(pq.ParquetFile(p).metadata.num_rows for p in Path(path).rglob("*.parquet"))
    summary["coco_pairs"] = {"rows": rows, "license": "not stated on the card (COCO captions CC BY 4.0; images from Flickr, per-image terms). Private research use only; do not redistribute.",
                             "source": "yhshin1020/coco-img-caption-pairs"}
    json.dump(summary, open(out / "blend_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
