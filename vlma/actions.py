"""操作文字列の書式と解析（R2-2, R2-3）。GenesisGeo の文字列と相互に変換する。

操作は2種類。
  CONSTRUCT e : perp a b c e [002] cong b c b e [003]
  APPLY Equal_sides_imply_equal_opposite_angles : eqangle b c c e c e b e [005] from [003]
前者は GenesisGeo の `<aux>` の1点分、後者は `<proof>` の1手に対応する。規則は、番号（r111）ではなく名前で書く（vlma.rules）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# 検証器が返す却下の理由。負例のラベルとして使う。
PARSE_FAILED = "parse_failed"  # 書式に合わない
BUILD_FAILED = "build_failed"  # 作図が構築できない
INVALID_STATEMENT = "invalid_statement"  # 結論の文が成り立たない形（未知の点・述語など）
NOT_HOLD = "not_hold"  # 結論が DDAR の導出結果に含まれない（偽の主張）
INVALID_CITATION = "invalid_citation"  # 根拠の番号が、まだ述べていない（存在しない）事実を指している
NOT_DERIVABLE = "not_derivable"  # 結論は真だが、書いた根拠の事実だけからは導けない（近道・飛躍）
UNKNOWN_RULE = "unknown_rule"  # 規則名が表にない、またはその規則はこの述語を結論にできない
RULE_MISMATCH = "rule_mismatch"  # 書いた根拠から結論は導けるが、書いた規則だけでは導けない（規則名が違う）
REJECT_REASONS = (PARSE_FAILED, BUILD_FAILED, INVALID_STATEMENT, NOT_HOLD, INVALID_CITATION, NOT_DERIVABLE,
                  UNKNOWN_RULE, RULE_MISMATCH)

_POINT = r"[a-z]\d*"
# 述語の引数は点、または定数（比 1/2、角 2pi/3 や 30o など）
_ARG = rf"(?:{_POINT}|\d[\w/.]*|pi[\w/.]*)"
_FACT = rf"[A-Za-z_]\w*(?: {_ARG})+ \[\d+\]"
_CONSTRUCT_RE = re.compile(rf"^CONSTRUCT (?P<point>{_POINT}) :(?P<body>(?: {_FACT})*)$")
_APPLY_RE = re.compile(
    rf"^APPLY (?P<rule>[A-Za-z]\w*) : (?P<pred>[A-Za-z_]\w*)(?P<args>(?: {_ARG})+)"
    r" \[(?P<id>\d+)\](?P<prem>(?: from(?: \[\d+\])+)?)$"
)
_IDS_RE = re.compile(r"\[(\d+)\]")


_FACT_RE = re.compile(rf"(?P<pred>[A-Za-z_]\w*)(?P<args>(?: {_ARG})+) \[(?P<id>\d+)\]")


def parse_facts(text: str) -> list[tuple[str, tuple[str, ...], str]]:
    """`perp a b c d [002] rconst a b a c 1/2 [003]` のような事実の並びを、(述語, 引数, 番号) の列にする。

    GenesisGeo の parse_fact_segments は、定数（`1/2` など）を引数に持つ事実を読み飛ばすため、こちらを使う。
    """
    return [(m["pred"], tuple(m["args"].split()), m["id"]) for m in _FACT_RE.finditer(normalize(text))]


def is_point(token: str) -> bool:
    return re.fullmatch(_POINT, token) is not None


@dataclass(frozen=True)
class Action:
    kind: str  # "construct" | "apply"
    point: str = ""  # construct: 新しい点の名前
    body: str = ""  # construct: 作図の条件（`perp a b c e [002] cong ...`）
    rule: str = ""  # apply: 規則名（r111, AR など）
    pred: str = ""  # apply: 結論の述語
    args: tuple[str, ...] = ()  # apply: 結論の引数（点）
    fact_id: str = ""  # apply: 結論に付く番号
    premises: tuple[str, ...] = ()  # apply: 根拠の番号

    def format(self) -> str:
        if self.kind == "construct":
            return f"CONSTRUCT {self.point} :" + (f" {self.body}" if self.body else "")
        text = f"APPLY {self.rule} : {self.pred} {' '.join(self.args)} [{self.fact_id}]"
        if self.premises:
            text += " from " + " ".join(f"[{p}]" for p in self.premises)
        return text

    def to_genesis(self) -> str:
        """GenesisGeo の文字列へ戻す。construct は `<aux>` の1点分、apply は `<proof>` の1手。"""
        if self.kind == "construct":
            return f"x00 {self.point} :" + (f" {self.body}" if self.body else "")
        from vlma.rules import rule_code

        text = f"{self.pred} {' '.join(self.args)} [{self.fact_id}] {rule_code(self.rule) or self.rule}"
        if self.premises:
            text += " " + " ".join(f"[{p}]" for p in self.premises)
        return text

    @property
    def point_args(self) -> tuple[str, ...]:
        """結論の引数のうち、点の名前であるもの（定数を除く）。"""
        return tuple(a for a in self.args if is_point(a))

    @property
    def conclusion_tokens(self) -> tuple[str, ...]:
        return (self.pred,) + self.args


@dataclass(frozen=True)
class ParseFailure:
    text: str
    reason: str = PARSE_FAILED


def normalize(text: str) -> str:
    return " ".join(str(text).split())


def parse_action(text: str) -> Action | ParseFailure:
    """モデルの出力文字列を操作に解析する。空白の違いは許し、それ以外の逸脱は解析失敗にする。"""
    t = normalize(text)
    m = _CONSTRUCT_RE.match(t)
    if m:
        return Action(kind="construct", point=m["point"], body=m["body"].strip())
    m = _APPLY_RE.match(t)
    if m:
        return Action(
            kind="apply",
            rule=m["rule"],
            pred=m["pred"],
            args=tuple(m["args"].split()),
            fact_id=m["id"],
            premises=tuple(_IDS_RE.findall(m["prem"])),
        )
    return ParseFailure(text=text)
