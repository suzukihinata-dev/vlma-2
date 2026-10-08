"""GenesisGeo の1問を、1手ごとのサンプルに展開し、元に戻す（R2-2）。

GenesisGeo の解析関数（parse_proof_step など）を再利用し、書式の変換だけをここで持つ。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from vlma.actions import Action, is_point, parse_facts
from vlma.rules import rule_name

_AUX_SEG_RE = re.compile(r"^x00\s+(?P<point>[a-z]\d*)\s*:\s*(?P<body>.*)$")


def problem_id(record: dict[str, Any]) -> str:
    return hashlib.sha1(record["fl_problem"].encode()).hexdigest()[:12]


def figure_path(pid: str, figure_index: int) -> str:
    """図の相対パス。作図が起きるたびに figure_index が1つ進み、同じ図は同じパスを指す。"""
    return f"figs/{pid}/fig{figure_index}.png"


def problem_points(llm_input: str) -> list[str]:
    """`<problem> a : ; b : perp ... ? goal </problem>` から、前提の点の名前を取る。"""
    content = _strip_tags(llm_input).split("?", 1)[0]
    points: list[str] = []
    for seg in content.split(";"):
        if ":" in seg:
            points.extend(seg.split(":", 1)[0].split())
    return points


def _strip_tags(text: str) -> str:
    return re.sub(r"</?\w+>", " ", text or "")


def _facts(section: str) -> list[tuple[str, tuple[str, ...], str]]:
    return parse_facts(_strip_tags(section))


def _gg():
    """GenesisGeo の解析関数。データを作るとき（to_compact など）だけ使う。学習側（expand）では使わない。"""
    from newclid.discovery.extraction import parsing

    return parsing


def _fact_text(fact: tuple[str, tuple[str, ...], str]) -> str:
    pred, args, fid = fact
    return f"{pred} {' '.join(args)} [{fid}]"


def extract_actions(record: dict[str, Any]) -> list[Action]:
    """`<aux>` の各点を construct、`<proof>` の各手を apply に変換する。作図が先、証明が後。"""
    out = record["llm_output_renamed"]
    actions: list[Action] = []
    for seg in _gg().extract_tag_content(out, "aux").split(";"):
        seg = " ".join(seg.split())
        if not seg:
            continue
        m = _AUX_SEG_RE.match(seg)
        if m is None:
            raise ValueError(f"unsupported aux segment: {seg!r}")
        actions.append(Action(kind="construct", point=m["point"], body=m["body"].strip()))
    for raw in _gg().extract_tag_content(out, "proof").split(";"):
        if not raw.strip():
            continue
        parsed = _gg().parse_proof_step(raw)
        if parsed is None:
            raise ValueError(f"unparsable proof step: {raw!r}")
        pred, args, fid, rule, prem = parsed
        actions.append(
            Action(kind="apply", rule=rule, pred=pred, args=tuple(args), fact_id=fid, premises=tuple(prem))
        )
    return actions


def _action_points(action: Action) -> list[str]:
    """手に関わる点。適用なら結論の点、作図なら新しい点と、その条件に出てくる点。"""
    if action.kind == "apply":
        return list(action.point_args)
    pts = [action.point]
    for _, args, _ in parse_facts(action.body):
        pts.extend(a for a in args if is_point(a))
    return pts


def to_compact(record: dict[str, Any]) -> dict[str, Any]:
    """1問を、保存用のコンパクトな形にする。1手ごとのサンプルは expand で作る。

    1手ごとに過去の操作列を持つと、サイズが手数の2乗で増えるため、保存はこの形で行う。
    """
    out = record["llm_output_renamed"]
    goal = _gg().extract_goal(record["llm_input_renamed"])
    return {
        "problem_id": problem_id(record),
        "fl_problem": record["fl_problem"],  # 前提の座標つき。検証と図の再現に必要
        "problem": record["llm_input_renamed"],
        "goal": f"{goal.predicate} {' '.join(goal.args)}" if goal else "",
        "actions": [a.format() for a in extract_actions(record)],
        "numerical_check": [_fact_text(f) for f in _facts(_gg().extract_tag_content(out, "numerical_check"))],
        "trivial": [_fact_text(f) for f in _facts(_gg().extract_tag_content(out, "trivial"))],
    }


def expand(compact: dict[str, Any]) -> list[dict[str, Any]]:
    """コンパクトな形から、1手ごとのサンプルに展開する。

    サンプルは入力（問題・目標・図・過去の操作列・情報）、thinking（空）、出力（次の1操作）、
    rejected（却下された出力。生成時は空）を持つ。
    情報は `numerical_check` と `trivial` のうち、その時点で存在する点だけを使う事実。
    """
    from dataclasses import replace

    from vlma.actions import parse_action
    from vlma.info import InfoLog

    # 案B: 証明が使う事実だけを選んだ numerical_check / trivial は、入力に使わない。
    # 手の根拠（from）に出てくるそれらの番号も外す。
    dropped = {f[2] for f in _facts(" ; ".join(compact["numerical_check"] + compact["trivial"]))}
    pid = compact["problem_id"]
    figure_index = 0
    history: list[str] = []
    samples: list[dict[str, Any]] = []
    info_log = InfoLog()

    for turn, raw in enumerate(compact["actions"], start=1):
        action = parse_action(raw)
        if not isinstance(action, Action):
            raise ValueError(f"unparsable action: {raw!r}")
        if action.kind == "apply":
            action = replace(action, premises=tuple(p for p in action.premises if p not in dropped),
                             rule=rule_name(action.rule, action.pred))  # 番号（r72）を名前に直す
        text = action.format()
        info = info_log.text()  # この手より前に適用した手に関わる数値
        samples.append(
            {
                "id": f"{pid}/{turn}",
                "problem_id": pid,
                "turn": turn,
                "problem": compact["problem"],
                "goal": compact["goal"],
                "figure_index": figure_index,
                "image": figure_path(pid, figure_index),
                "history": list(history),
                "info": info,
                "thinking": "",
                "output": text,
                "rejected": [],
            }
        )
        history.append(text)
        if action.kind == "construct":
            figure_index += 1
        fig_points = compact.get("figure_points")
        if fig_points is not None:  # この手に関わる点の数値を、次の手以降の情報欄に追記する
            info_log.add(fig_points[figure_index], _action_points(action))
    return samples


def to_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    """1問を手数分のサンプルに展開する（to_compact と expand の組み合わせ）。"""
    return expand(to_compact(record))


def from_steps(samples: list[dict[str, Any]]) -> str:
    """サンプル列から、GenesisGeo の `<aux>` と `<proof>` 相当の文字列を作る（往復確認用）。

    numerical_check と trivial は入力に使わないので戻らない。手の根拠からも、それらの番号は外れている。
    """
    from vlma.actions import parse_action

    aux, proof = [], []
    for s in samples:
        a = parse_action(s["output"])
        if not isinstance(a, Action):
            raise ValueError(f"unparsable output: {s['output']!r}")
        (aux if a.kind == "construct" else proof).append(a.to_genesis() + " ;")
    parts = []
    if aux:
        parts.append("<aux> " + " ".join(aux) + " </aux>")
    parts.append("<proof> " + " ".join(proof) + " </proof>")
    return " ".join(parts)
