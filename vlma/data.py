"""学習用データセット（方式 A'）。読み込むときに、1問を1手ごとのサンプルに展開する。

方式 A': ターンごとに独立したサンプル。画像は「最初のターンと、作図の手のあと（図が変わったターン）」だけ入れる。
サンプルの文章:
  [<|vision_start|><|image_pad|><|vision_end|>\\n]
  <problem> … </problem>\\n
  [<info> … </info>\\n]
  [<history>\\n 過去の手（1行に1手）\\n</history>\\n]
  <think>\\n\\n</think>\\n\\n        （thinking の枠。中身は空）
  出力（次の1手）<|im_end|>          ← 損失をかけるのは、この出力と終わりの記号だけ
学習するターンは、既定では、各問題の全ターン。量は、問題数で絞る（max_problems、または問題の一覧 subset）。
turns_per_problem を整数にすると、1問あたりそのターン数に絞れる（作図の手は必ず入れる。速度の測定などの試験用）。

torch と PIL は、データを取り出すときに読み込む。プロンプトの組み立て（build_text）は、それらなしで試せる。
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from vlma.steps import expand

IMAGE_MARK = "<|vision_start|><|image_pad|><|vision_end|>"
END = "<|im_end|>"


def shows_image(samples: list[dict[str, Any]], k: int) -> bool:
    """k 番目（0 始まり）のターンで画像を入れるか。最初のターンと、作図の手のあと。"""
    return k == 0 or samples[k - 1]["output"].startswith("CONSTRUCT")


def build_text(sample: dict[str, Any], with_image: bool) -> tuple[str, str]:
    """(入力の文章, 出力の文章) を返す。出力の文章に、終わりの記号を含む。"""
    parts = []
    if with_image:
        parts.append(IMAGE_MARK + "\n")
    parts.append(sample["problem"] + "\n")
    if sample["info"]:
        parts.append("<info> " + sample["info"] + " </info>\n")
    if sample["history"]:
        parts.append("<history>\n" + "\n".join(sample["history"]) + "\n</history>\n")
    parts.append("<think>\n\n</think>\n\n")
    return "".join(parts), sample["output"] + END


THINK = "<think>\n\n</think>\n\n"
_INFO_ITEM = re.compile(r"\S+ \S+ ;")
_INFO_SECTION = re.compile(r"<(lengths|angles)> (.*?) </\1>")


def info_delta(prev: str, cur: str) -> str:
    """情報欄 cur に、prev から新しく加わった数値だけを、同じ形（<lengths>…</lengths> <angles>…</angles>）で返す。"""
    old = set(_INFO_ITEM.findall(prev))
    parts = []
    for tag, body in _INFO_SECTION.findall(cur):
        new = [x for x in _INFO_ITEM.findall(body) if x not in old]
        if new:
            parts.append(f"<{tag}> " + " ".join(new) + f" </{tag}>")
    return " ".join(parts)


def build_sequence(samples: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """方式 B: 1問の全ターンを1本の系列にする。(全文, 画像のパスの列) を返す。損失をかける位置は、各ターンの出力と終わりの記号。

    ターン k の入力は、画像（最初と、作図の手のあと）、問題（最初だけ）、情報欄に新しく加わった数値、空の thinking の枠。
    過去の手は系列の中にあるので、<history> は書かない。
    """
    parts, images = [], []
    prev_info = ""
    for k, sm in enumerate(samples):
        if shows_image(samples, k):
            parts.append(IMAGE_MARK + "\n")
            images.append(sm["image"])
        if k == 0:
            parts.append(sm["problem"] + "\n")
        delta = info_delta(prev_info, sm["info"])
        if delta:
            parts.append("<info> " + delta + " </info>\n")
        prev_info = sm["info"]
        parts.append(THINK + sm["output"] + END)
    return "".join(parts), images


def target_mask(input_ids, think_end: int, im_end: int, n_turns: int):
    """系列のトークン列で、損失をかける位置（各ターンの、</think> と改行のあとから <|im_end|> まで）を True にした真偽値の列を返す。"""
    ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    mask = [False] * len(ids)
    spans, i = 0, 0
    while i < len(ids):
        if ids[i] == think_end:
            j = ids.index(im_end, i + 2)  # i+1 は改行、i+2 から出力
            for t in range(i + 2, j + 1):
                mask[t] = True
            spans, i = spans + 1, j
        i += 1
    if spans != n_turns:
        raise ValueError(f"found {spans} output spans, expected {n_turns}")
    return mask


def choose_turns(n_turns: int, construct: list[int], k: int | None, rng: random.Random) -> list[int]:
    """k が None なら全ターン。整数なら、作図の手は全て入れ、残りから、合計が k 個になるまで選ぶ（k 個以上の作図があれば、作図だけ）。"""
    if k is None:
        return list(range(n_turns))
    cs = set(construct)
    rest = [t for t in range(n_turns) if t not in cs]
    rng.shuffle(rest)
    return sorted(construct + rest[: max(0, k - len(construct))])


def build_index(data_dir: str | Path) -> Path:
    """problems.jsonl の各行について、位置・ターン数・作図の手の位置を求めて、turn_index.json に保存する（一度だけ行う）。"""
    d = Path(data_dir)
    index = {}
    with open(d / "problems.jsonl", "rb") as f:
        pos = 0
        for line in f:
            c = json.loads(line)
            index[c["problem_id"]] = [pos, len(c["actions"]), [t for t, a in enumerate(c["actions"]) if a.startswith("CONSTRUCT")]]
            pos += len(line)
    out = d / "turn_index.json"
    json.dump(index, open(out, "w"))
    return out


class VlmaDataset:
    """(問題, ターン) の列。epoch ごとに set_epoch で、選ぶターンを変える。

    data_dir に problems.jsonl（作成済みのコンパクトな形）、split.json（問題 id → train / val / test / excluded）、
    turn_index.json（build_index で作る）を置く。
    """

    def __init__(self, data_dir: str | Path, split: str, processor, *, turns_per_problem: int | None = None, seed: int = 0,
                 max_problems: int | None = None, subset: str | Path | None = None, max_length: int = 8192):
        self.dir = Path(data_dir)
        self.processor = processor
        self.k = turns_per_problem
        self.seed = seed
        self.max_length = max_length
        if not (self.dir / "turn_index.json").exists():
            build_index(self.dir)
        meta = json.load(open(self.dir / "turn_index.json"))
        split_map = json.load(open(self.dir / "split.json"))
        self.problems = [(pid, *m) for pid, m in meta.items() if split_map.get(pid) == split]
        if subset is not None:  # 学習に使う問題の一覧（問題 id の JSON の配列）。order は一覧に従う
            wanted = {pid: n for n, pid in enumerate(json.load(open(subset)))}
            self.problems = sorted((p for p in self.problems if p[0] in wanted), key=lambda p: wanted[p[0]])
        if max_problems:
            self.problems = self.problems[:max_problems]
        self._f = None
        self.set_epoch(0)

    def _problem(self, offset: int) -> dict[str, Any]:
        if self._f is None:
            self._f = open(self.dir / "problems.jsonl", "rb")
        self._f.seek(offset)
        return json.loads(self._f.readline())

    def set_epoch(self, epoch: int) -> None:
        """この epoch で使う（問題の番号, ターン）の一覧を作る。選び方は seed・epoch・問題 id で決まる。"""
        self.epoch = epoch
        self.index = [
            (i, t)
            for i, (pid, _, n, construct) in enumerate(self.problems)
            for t in choose_turns(n, construct, self.k, random.Random(f"{self.seed}:{epoch}:{pid}"))
        ]

    def __len__(self) -> int:
        return len(self.index)

    def example(self, i: int, turn: int) -> dict[str, Any]:
        """1サンプルぶんの文章・画像の情報を返す（トークン化の前）。"""
        samples = expand(self._problem(self.problems[i][1]))
        with_image = shows_image(samples, turn)
        prompt, target = build_text(samples[turn], with_image)
        image = str(self.dir / samples[turn]["image"]) if with_image else None
        return {"prompt": prompt, "target": target, "image": image, "id": samples[turn]["id"]}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        i, turn = self.index[idx]
        ex = self.example(i, turn)
        images = [Image.open(ex["image"]).convert("RGB")] if ex["image"] else None
        enc = self.processor(text=[ex["prompt"] + ex["target"]], images=images, return_tensors="pt")
        n_prompt = self.processor(text=[ex["prompt"]], images=images, return_tensors="pt")["input_ids"].shape[1]
        input_ids = enc["input_ids"][0]
        if len(input_ids) > self.max_length:
            raise ValueError(f"sample {ex['id']} has {len(input_ids)} tokens > {self.max_length}")
        labels = input_ids.clone()
        labels[:n_prompt] = -100  # 損失は、出力と終わりの記号だけ
        item = {"input_ids": input_ids, "labels": labels, "attention_mask": enc["attention_mask"][0], "id": ex["id"]}
        for k in ("pixel_values", "image_grid_thw"):
            if k in enc:
                item[k] = enc[k]
        if "mm_token_type_ids" in enc:  # 画像・文章の位置の種類（M-RoPE の計算に要る）。画像なしのサンプルにも付く
            item["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
        return item


class VlmaSeqDataset(VlmaDataset):
    """方式 B: 1問 = 1本の系列（全ターンを続ける）。損失は全ターンの出力にかかる。他は VlmaDataset と同じ（split、subset、max_problems）。"""

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.index = [(i, 0) for i in range(len(self.problems))]

    def example(self, i: int, turn: int = 0) -> dict[str, Any]:
        samples = expand(self._problem(self.problems[i][1]))
        text, images = build_sequence(samples)
        return {"text": text, "images": [str(self.dir / p) for p in images], "n_turns": len(samples), "id": samples[0]["problem_id"]}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import torch
        from PIL import Image

        i, _ = self.index[idx]
        ex = self.example(i)
        images = [Image.open(p).convert("RGB") for p in ex["images"]] or None
        enc = self.processor(text=[ex["text"]], images=images, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        if len(input_ids) > self.max_length:
            raise ValueError(f"problem {ex['id']} has {len(input_ids)} tokens > {self.max_length}")
        tok = self.processor.tokenizer
        mask = torch.tensor(target_mask(input_ids, tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>"), ex["n_turns"]))
        labels = torch.where(mask, input_ids, torch.full_like(input_ids, -100))
        item = {"input_ids": input_ids, "labels": labels, "attention_mask": enc["attention_mask"][0], "id": ex["id"]}
        for k in ("pixel_values", "image_grid_thw"):
            if k in enc:
                item[k] = enc[k]
        if "mm_token_type_ids" in enc:
            item["mm_token_type_ids"] = enc["mm_token_type_ids"][0]
        return item
