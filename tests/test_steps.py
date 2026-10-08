import json
import re
from pathlib import Path

import pytest

from vlma.actions import Action, parse_action
from vlma.steps import from_steps, to_steps

RECORDS = json.loads((Path(__file__).parent / "data" / "records.json").read_text())


def norm(s):
    return " ".join(s.split())


def without_unused_sections(rec):
    """元の出力から、numerical_check と trivial の節と、それらの番号を取り除いたもの（案B で入力に使わない部分）。"""
    out = rec["llm_output_renamed"]
    ids = set()
    for tag in ("numerical_check", "trivial"):
        m = re.search(rf"<{tag}>(.*?)</{tag}>", out, re.S)
        if m:
            ids |= set(re.findall(r"\[(\d+)\]", m.group(1)))
            out = out.replace(m.group(0), "")
    for i in ids:
        out = out.replace(f" [{i}]", "")
    return norm(out)


@pytest.mark.parametrize("i", range(len(RECORDS)))
def test_roundtrip_restores_aux_and_proof(i):
    rec = RECORDS[i]
    assert from_steps(to_steps(rec)) == without_unused_sections(rec)


def test_first_record_turns_and_order():
    steps = to_steps(RECORDS[0])
    assert len(steps) == 13  # 作図1手 + 証明12手
    kinds = [parse_action(s["output"]).kind for s in steps]
    assert kinds == ["construct"] + ["apply"] * 12  # 作図が先
    assert [s["turn"] for s in steps] == list(range(1, 14))
    assert steps[0]["history"] == [] and steps[1]["history"] == [steps[0]["output"]]
    assert all(s["thinking"] == "" and s["rejected"] == [] for s in steps)
    assert steps[0]["goal"] == "eqangle a b a c b d c d"


def test_numeric_fact_ids_are_not_cited():
    from vlma.steps import to_compact

    for rec in RECORDS:
        c = to_compact(rec)
        ids = set(re.findall(r"\[(\d+)\]", " ".join(c["numerical_check"] + c["trivial"])))
        assert ids  # この2件は、どちらも数値的な事実を持つ
        cited = " ".join(s["output"] for s in to_steps(rec))
        assert not any(f"[{i}]" in cited for i in ids)
        # 元の手には番号が出てくる（外したことの確認）
        assert any(f"[{i}]" in " ".join(c["actions"]) for i in ids)


def test_info_is_appended_after_each_step_from_the_points_it_involves():
    from vlma.info import InfoLog
    from vlma.steps import expand, to_compact

    c = to_compact(RECORDS[0])
    pts = {"a": [0.0, 0.0], "b": [3.0, 0.0], "c": [0.0, 4.0], "d": [1.0, 1.0], "e": [2.0, 2.0]}
    assert all(s["info"] == "" for s in expand(c))  # 座標が無ければ空
    c["figure_points"] = [pts, pts]
    steps = expand(c)
    assert steps[0]["info"] == ""  # 最初の手の前には、何も追記されていない
    # 作図 CONSTRUCT e : perp a b c e ... の後は、点 e と、条件に出てくる a, b, c の数値が入る
    assert "<lengths>" in steps[1]["info"] and "ae" in steps[1]["info"] and "bc 5.0000" in steps[1]["info"]
    assert "dc" not in steps[1]["info"]  # 手に関わらない点 d は入らない
    # 追記は増えるだけで、同じ数値は繰り返さない
    lens = [s["info"] for s in steps]
    assert all(len(lens[i]) <= len(lens[i + 1]) for i in range(len(lens) - 1))
    assert steps[-1]["info"].count("bc 5.0000") == 1
    # 三角形（3点）には、三辺と三つの角が入る
    log = InfoLog()
    log.add(pts, ["a", "b", "c"])
    t = log.text()
    assert "ab 3.0000" in t and "ac 4.0000" in t and "bc 5.0000" in t
    assert "bac 90.0" in t and "abc 53.1" in t and "acb 36.9" in t
    # 5点以上の手は、角度を載せない（長さだけ）
    big = InfoLog()
    big.add({**pts, "f": [5.0, 5.0]}, ["a", "b", "c", "d", "e"])
    assert "<angles>" not in big.text() and "<lengths>" in big.text()


def test_figure_changes_only_on_construct():
    rec = RECORDS[1]
    steps = to_steps(rec)
    n_construct = sum(parse_action(s["output"]).kind == "construct" for s in steps)
    assert n_construct == 2
    assert [s["figure_index"] for s in steps][:3] == [0, 1, 2]
    # 同じ図を指すターンは同じパス。作図のたびに新しいパスになる
    paths = [s["image"] for s in steps]
    assert paths[0] == f"figs/{steps[0]['problem_id']}/fig0.png"
    assert [p for p in paths if p.endswith("fig1.png")] and [p for p in paths if p.endswith("fig2.png")]
    assert paths[0] != paths[1] and paths[1] != paths[2] and paths[2] == paths[3]
    assert len({s["figure_index"] for s in steps}) == 3


@pytest.mark.parametrize("i", range(len(RECORDS)))
def test_compact_roundtrip_and_size(i):
    from vlma.steps import expand, to_compact

    rec = RECORDS[i]
    c = to_compact(rec)
    assert json.loads(json.dumps(c, ensure_ascii=False)) == c  # JSON にできる
    assert expand(c) == to_steps(rec)
    # コンパクトな形は、展開した形よりずっと小さい（手数の2乗を避ける）
    assert len(json.dumps(c)) < len(json.dumps(expand(c))) / 2
