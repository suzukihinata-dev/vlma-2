import pytest

from vlma import rules

# 保存した証明（10万問・680万手）に出てきた (番号, 結論の述語) の全66通り
OBSERVED = """
AR eqangle|AR eqratio|AR cong|AR para|AR perp|AR coll|AR constline|AR rconst
Transfer eqratio|Transfer cong|Transfer para|Transfer eqangle|Transfer simtrir|Transfer simtri|Transfer perp|Transfer coll
Transfer contrir|Transfer contri|Transfer cyclic|Transfer midp|Transfer rconst
r03 eqangle|r04 cyclic|r07 eqratio|r103 cong|r103 eqangle|r104 eqangle|r104 cong|r105 eqratio|r107 eqpoint|r108 constline
r108 cong|r109 eqpoint|r11 eqangle|r110 cong|r110 contri|r111 eqangle|r111 contrir|r112 contri|r113 contrir|r12 eqratio
r19 cong|r27 para|r28 coll|r34 simtri|r35 simtrir|r41 para|r42 eqratio|r43 perp|r46 eqangle|r49 cong|r50 cong|r51 rconst
r52 eqratio|r52 eqangle|r53 eqratio|r53 eqangle|r54 midp|r56 coll|r56 cong|r60 simtri|r61 simtrir|r62 simtri|r63 simtrir
r72 circle|r82 para
"""
PAIRS = [tuple(x.split()) for chunk in OBSERVED.split("\n") for x in chunk.split("|") if x.strip()]


def test_all_observed_pairs_have_a_name():
    assert len(PAIRS) == 66
    for code, pred in PAIRS:
        name = rules.rule_name(code, pred)
        assert rules.is_valid(name, pred) and rules.rule_code(name) == code


def test_the_two_meanings_of_r110_and_r111_get_different_names():
    assert rules.rule_name("r111", "eqangle") == "Equal_sides_imply_equal_opposite_angles"
    assert rules.rule_name("r111", "contrir") == "SSS_congruence_of_triangles_reverse"
    assert rules.rule_name("r110", "cong") != rules.rule_name("r110", "contri")
    assert rules.rule_name("r28", "coll") != rules.rule_name("r82", "para")  # 公式の名前は重複している


def test_names_are_unique_and_plain_words():
    assert len(set(rules.ALL_NAMES)) == len(rules.ALL_NAMES)
    assert all(n.replace("_", "").isalnum() and n[0].isalpha() for n in rules.ALL_NAMES)


def test_unknown_names_and_mismatched_conclusions_are_invalid():
    assert rules.rule_code("r72") is None and not rules.is_valid("r72", "circle")
    assert not rules.is_valid("Thales_theorem_I", "cong")
    assert rules.is_valid("Angle_chasing", "para") and not rules.is_valid("Angle_chasing", "cong")
    # AR の種類は C++ 版 DDAR の分類に合わせてある: Slope=角度、Dist=長さ、Distlog=比。coll と rconst は Dist
    assert rules.is_valid("Angle_chasing", "constline") and not rules.is_valid("Angle_chasing", "coll")
    assert rules.is_valid("Length_chasing", "coll") and rules.is_valid("Length_chasing", "rconst")
    assert rules.is_valid("Ratio_chasing", "eqratio") and not rules.is_valid("Ratio_chasing", "rconst")
    assert rules.is_valid("Length_chasing", "cong") and rules.is_valid("Ratio_chasing", "cong")  # cong は両方で出る
    assert rules.rule_name("AR", "cong") == "Length_chasing" and rules.rule_name("AR", "eqratio") == "Ratio_chasing"
    with pytest.raises(ValueError):
        rules.rule_name("r72", "cong")


def test_to_genesis_goes_back_to_the_number():
    from vlma.actions import parse_action

    a = parse_action("APPLY Equal_distances_define_circumcenter : circle e d f h [010] from [002] [009]")
    assert a.to_genesis() == "circle e d f h [010] r72 [002] [009]"
