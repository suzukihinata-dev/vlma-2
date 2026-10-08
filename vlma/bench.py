"""ベンチマークのモデル側: 環境の状態からプロンプトを作る（学習データと同じ形）と、結果の集計。torch も newclid も要らない。

方式ごとのプロンプト（学習データ vlma.data と同じ。環境の状態 state は vlma.env.Episode.state）:
  TurnPrompt  方式 A': ターンごとに、問題・情報欄・過去の手（<history>）を書き直す。
  SeqPrompt   方式 B: 受理した手を、1本の系列に続ける。毎ターン、新しい画像（作図のあと）と、新しく加わった数値だけを足す。
どちらも、prompt(state) は状態を変えない（却下された手を、もう一度出し直すとき、同じプロンプトになる）。
受理した手だけを accept(state, output) で足す。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from vlma.data import END, IMAGE_MARK, THINK, build_text, info_delta


class TurnPrompt:
    def prompt(self, state: dict[str, Any]) -> tuple[str, list[str]]:
        text, _ = build_text({"problem": state["problem"], "info": state["info"], "history": state["history"], "output": ""},
                             with_image=state["image"] is not None)
        return text, [state["image"]] if state["image"] else []

    def accept(self, state: dict[str, Any], output: str) -> None:
        pass


class SeqPrompt:
    def __init__(self) -> None:
        self.text = ""
        self.images: list[str] = []
        self.prev_info = ""
        self.first = True

    def _pending(self, state: dict[str, Any]) -> str:
        parts = []
        if state["image"]:
            parts.append(IMAGE_MARK + "\n")
        if self.first:
            parts.append(state["problem"] + "\n")
        delta = info_delta(self.prev_info, state["info"])
        if delta:
            parts.append("<info> " + delta + " </info>\n")
        parts.append(THINK)
        return "".join(parts)

    def prompt(self, state: dict[str, Any]) -> tuple[str, list[str]]:
        return self.text + self._pending(state), self.images + ([state["image"]] if state["image"] else [])

    def accept(self, state: dict[str, Any], output: str) -> None:
        self.text += self._pending(state) + output + END
        if state["image"]:
            self.images.append(state["image"])
        self.prev_info, self.first = state["info"], False


FORMATS = {"turn": TurnPrompt, "seq": SeqPrompt}


def clean_output(text: str) -> str:
    """生成した文章から、終わりの記号などを外して、1手の文字列にする。"""
    return text.split(END)[0].replace("<|endoftext|>", "").strip()


def _rate(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """エピソードの結果（vlma.env.Episode.result）の一覧を、指標にまとめる。"""
    n = len(results)
    kinds = Counter(r["kind"] for r in results)
    reasons: Counter = Counter()
    for r in results:
        reasons.update(r["reject_reasons"])
    n_rej = sum(r["rejected"] for r in results)
    n_att = sum(r["attempts"] for r in results)
    bins: dict[str, list] = defaultdict(list)
    for r in results:
        k = r["n_reference_turns"]
        bins["turns<=20" if k <= 20 else "turns 21-50" if k <= 50 else "turns 51-100" if k <= 100 else "turns>100"].append(r)
    solved = [r for r in results if r["success"]]
    return {
        "problems": n,
        "success_rate": _rate(len(solved), n),
        "kind_rate": {k: _rate(kinds[k], n) for k in ("shortest", "wasteful", "incomplete")},
        "move_accept_rate": _rate(n_att - n_rej, n_att),
        "attempts_per_problem": round(n_att / n, 2) if n else 0,
        "reject_reasons": dict(reasons.most_common()),
        "first_reject_at_attempt_median": sorted(r["first_reject_attempt"] for r in results if r["first_reject_attempt"] is not None)[
            sum(r["first_reject_attempt"] is not None for r in results) // 2] if any(r["first_reject_attempt"] is not None for r in results) else None,
        "progress": round(sum(min(1.0, r["accepted"] / r["n_reference_turns"]) for r in results) / n, 4) if n else 0,
        # 参照証明が使う事実（適用の手の結論）のうち、予算の中で見つけた割合の平均。目標に届かなくても、必要な事実をどこまで見つけたか
        "reference_coverage": round(sum(min(1.0, r["needed"] / r["n_reference"]) for r in results if r["n_reference"]) / n, 4) if n else 0,
        # 参照の手数に対する、試行（却下を含む）の倍率の中で、目標に届いた割合
        "success_within": {f"{m}x_reference_turns": _rate(sum(r["success"] and r["attempts"] <= m * r["n_reference_turns"] for r in results), n)
                           for m in (1, 1.5, 2)},
        "needed_share_of_accepted_apply": _rate(sum(r["needed"] for r in results), sum(r["needed"] + r["unneeded"] for r in results)),
        "reward_mean": round(sum(r["reward_total"] for r in results) / n, 4) if n else 0,
        "success_by_reference_length": {k: {"n": len(v), "success_rate": _rate(sum(x["success"] for x in v), len(v))} for k, v in sorted(bins.items())},
        "solved_length_vs_reference": round(sum(r["accepted"] for r in solved) / sum(r["n_reference_turns"] for r in solved), 3) if solved else None,
    }
