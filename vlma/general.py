"""忘却の確認に使う、一般文章。学習には使わない。

wikitext-103 のテスト分（Wikipedia の文章）を、決まった長さのトークンに区切る。
一般文章の perplexity を、学習の前後で比べて、元の知識がどれだけ保たれているかを見る。
注意: Wikipedia だけの代用で、「一般的な知識」全体を表すものではない。
"""
from __future__ import annotations

import torch


def load_general_text(n_chunks: int, tokenizer, chunk_tokens: int = 1024) -> list[torch.Tensor]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", "wikitext-103-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    chunks = [torch.tensor(ids[i : i + chunk_tokens]) for i in range(0, len(ids) - chunk_tokens, chunk_tokens)]
    return chunks[:n_chunks]
