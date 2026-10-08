import json
from pathlib import Path

import pytest

from vlma import actions as A
from vlma.figures import find_placement_seed
from vlma.steps import to_steps
from vlma.verify import StepVerifier

RECORDS = json.loads((Path(__file__).parent / "data" / "records.json").read_text())


@pytest.mark.parametrize("engine", ["c", "python", "both"])
def test_reference_sequence_is_accepted_and_goal_only_at_last(engine):
    rec = RECORDS[0]
    steps = to_steps(rec)
    v = StepVerifier(rec, engine=engine)
    verdicts = [v.check(s["output"]) for s in steps]
    assert all(x.ok for x in verdicts)
    assert [i for i, x in enumerate(verdicts) if x.goal_reached] == [len(steps) - 1]


def test_rejection_reasons():
    rec = RECORDS[0]
    steps = to_steps(rec)
    v = StepVerifier(rec)
    assert v.check("hello").reason == A.PARSE_FAILED
    assert v.check("CONSTRUCT z : foo a b z [900]").reason == A.BUILD_FAILED
    assert v.check(steps[0]["output"]).ok  # 作図を受理
    assert v.check("CONSTRUCT e : perp a b c e [002]").reason == A.BUILD_FAILED  # 点 e は既にある
    assert v.check("APPLY Length_chasing : cong a b a q9 [900]").reason == A.INVALID_STATEMENT  # 未知の点
    assert v.check("APPLY Length_chasing : cong a b a c [900]").reason == A.NOT_HOLD  # 成り立たない結論


def _verifier_after_constructs(rec, **kw):
    steps = to_steps(rec)
    v = StepVerifier(rec, reference=[s["output"] for s in steps], **kw)
    for s in steps:
        if s["output"].startswith("CONSTRUCT"):
            assert v.check(s["output"]).ok
    return v, steps


def test_goal_cannot_be_stated_directly_after_the_constructs():
    """作図のあとに目標の文をそのまま出す近道は、書いた根拠から導けないので却下される。"""
    rec = RECORDS[0]
    v, _ = _verifier_after_constructs(rec)
    assert v.check("APPLY Angle_chasing : eqangle a b a c b d c d [900] from [000]").reason == A.NOT_DERIVABLE
    # 前提を全部書いても、1段では導けない
    assert v.check("APPLY Angle_chasing : eqangle a b a c b d c d [900] from [000] [001] [002] [003]").reason == A.NOT_DERIVABLE


def test_rule_must_be_a_known_name_that_can_conclude_the_statement():
    v, steps = _verifier_after_constructs(RECORDS[0])
    first = steps[1]["output"]
    body = first.split(" : ", 1)[1]
    assert v.check("APPLY r111 : " + body).reason == A.UNKNOWN_RULE  # 番号は受け付けない
    assert v.check("APPLY Made_up_theorem : " + body).reason == A.UNKNOWN_RULE
    assert v.check("APPLY Thales_theorem_I : " + body).reason == A.UNKNOWN_RULE  # eqangle は結論にできない規則
    assert v.check(first).ok  # 正しい名前なら通る


def test_named_rule_must_be_able_to_derive_the_step_from_the_cited_facts():
    """規則名がその手で使えるか。書いた規則だけを有効にした DDAR で導けなければ rule_mismatch。"""
    v, steps = _verifier_after_constructs(RECORDS[0])
    first = steps[1]["output"]
    assert " : " in first and first.startswith("APPLY Equal_sides_imply_equal_opposite_angles")
    body = first.split(" : ", 1)[1]
    # 角度の代数的推論では、等しい辺から底角が等しいことは導けない
    assert v.check("APPLY Angle_chasing : " + body).reason == A.RULE_MISMATCH
    # 別の幾何の定理でも導けない
    assert v.check("APPLY Arc_determines_internal_angles : " + body).reason == A.RULE_MISMATCH
    assert v.check(first).ok  # 正しい規則名なら通る


@pytest.mark.parametrize("i", range(len(RECORDS)))
def test_every_stored_proof_is_accepted_with_rule_names_checked(i):
    """保存した正しい証明は、規則名の確認を含めて、全手が受理される（誤却下がない）。

    RECORDS[3] は、Thales_theorem_I が導く形を AR で並べ替えた結論を含む（規則だけでは導けず、規則と AR で導ける手）。
    """
    from vlma.reward import step_rewards

    rec = RECORDS[i]
    steps = to_steps(rec)
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    verdicts = [v.check(s["output"]) for s in steps]
    assert all(x.ok for x in verdicts), [(k + 1, x.reason, steps[k]["output"][:80]) for k, x in enumerate(verdicts) if not x.ok][:3]
    assert verdicts[-1].goal_reached
    assert step_rewards(verdicts, sum(s["output"].startswith("APPLY") for s in steps))[1] == "shortest"


def test_transfer_must_cite_an_eqpoint_fact():
    v, steps = _verifier_after_constructs(RECORDS[0])
    first = steps[1]["output"]
    body = first.split(" : ", 1)[1]
    # Transfer は、同じ位置の2点（eqpoint）の間で事実を置き換える仕組み。根拠に eqpoint がなければ、規則の取り違え
    assert v.check("APPLY Transfer_between_equal_points : " + body).reason == A.RULE_MISMATCH


def test_cited_fact_must_already_be_stated():
    v, steps = _verifier_after_constructs(RECORDS[0])
    first = steps[1]["output"]  # 最初の適用の手
    assert v.check(first.replace("from [003]", "from [999]")).reason == A.INVALID_CITATION
    # まだ述べていない（これから出てくる）手の番号も、根拠にできない
    assert v.check(first.replace("from [003]", "from [016]")).reason == A.INVALID_CITATION


def test_restating_a_known_fact_is_accepted_but_not_needed():
    v, steps = _verifier_after_constructs(RECORDS[0])
    first = steps[1]["output"]
    assert v.check(first).needed is True
    again = v.check(first.replace("[005]", "[901]"))
    assert again.ok and again.needed is False


def test_reference_sequence_gets_strong_rewards_and_a_wasteful_one_does_not():
    from vlma.reward import RewardConfig, step_rewards

    rec = RECORDS[0]
    steps = to_steps(rec)
    n_ref = sum(s["output"].startswith("APPLY") for s in steps)
    cfg = RewardConfig()
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    verdicts = [v.check(s["output"]) for s in steps]
    assert all(x.ok for x in verdicts) and verdicts[-1].goal_reached
    rewards, kind = step_rewards(verdicts, n_ref, cfg)
    assert kind == "shortest" and rewards[0] == 0.0 and all(r == cfg.strong for r in rewards[1:])
    # 言い直しを1手足すと、導出は無駄のあるものになり、足した手は負の報酬、必要な手は弱い報酬になる
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    outs = [s["output"] for s in steps]
    outs.insert(3, outs[2].replace("[006]", "[902]"))
    verdicts = [v.check(o) for o in outs]
    rewards, kind = step_rewards(verdicts, n_ref, cfg)
    assert kind == "wasteful"
    assert rewards[3] == cfg.negative and rewards[1] == cfg.weak and rewards[-1] == cfg.weak
    # 目標に届かなければ、必要な手の報酬は0
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    verdicts = [v.check(o) for o in [s["output"] for s in steps][:5]]
    rewards, kind = step_rewards(verdicts, n_ref, cfg)
    assert kind == "incomplete" and set(rewards) == {0.0}
    # 却下された手は rejected
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    verdicts = [v.check("hello")]
    assert step_rewards(verdicts, n_ref, cfg)[0] == [cfg.rejected]


def test_circle_conclusion_is_checked_against_its_cited_facts():
    """circle を結論とする手も、書いた根拠から導けるかを確認する（例外にしない）。"""
    rec = RECORDS[2]  # 規則 r72（cong, cong => circle）の手を含む
    steps = to_steps(rec)
    circle_steps = [s["output"] for s in steps if "circle" in s["output"]]
    assert circle_steps
    v = StepVerifier(rec, reference=[s["output"] for s in steps])
    verdicts = [v.check(s["output"]) for s in steps]
    assert all(x.ok for x in verdicts)  # 保存した証明は通る
    v, _ = _verifier_after_constructs(rec)
    out = v.check("APPLY Equal_distances_define_circumcenter : circle e d f h [900] from [000]")  # 無関係な根拠で circle を主張する近道
    assert out.reason == A.NOT_DERIVABLE


@pytest.mark.parametrize("i", range(len(RECORDS)))
def test_placement_seed_exists(i):
    assert find_placement_seed(RECORDS[i]) is not None
