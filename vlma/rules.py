"""規則の名前。手の出力では、GenesisGeo の規則の番号（r72 など）の代わりに、意味のある名前を使う。

番号は意味を持たず、一意でもない（r110・r111 は、結論の述語によって別の定理を指す）。名前は公式の名前
（GenesisGeo の rules.txt と、C++ 版の DDAR/theorem.cpp）に従う。次の場合だけ、こちらで決めた。
  - 公式の名前が重複する: r28 と r82 はどちらも "Overlapping parallels"。r82（共線 → 平行）を Collinear_implies_parallel とした。
  - 直接（同じ向き）と逆（逆向き）の違いは、末尾に direct / reverse を付けた。
  - AR（代数的推論）は、結論の述語で3種類に分けた。C++ 版の "AR with Slope / Dist / Distlog" に対応する（下の表を参照）。
    競技の幾何でよく使う名前（Angle_chasing など）にした。
  - 公式の名前が定義の内容と合わない、または意味が取りにくいものは、定義（前提 ⇒ 結論）に合わせて直した:
    r49, r50, r72, r73, r108, r109。対応は docs/R2_RULE_NAMES.md。
  - Transfer は C++ 版の名前そのままでは意味が取れないので、Transfer_between_equal_points とした。
保存したデータ（compact）は、GenesisGeo の番号のまま持つ。名前に直すのは expand のとき。
"""
from __future__ import annotations

# (番号, 結論にできる述語の集合 または None=制限しない, 名前)
# 述語の集合は、保存した証明の680万手で観測した (番号, 結論の述語) の組から決めた。データに出てこない規則は None。
_ENTRIES: list[tuple[str, frozenset[str] | None, str]] = [
    # AR の種類は、C++ 版 DDAR が付ける理由（AR with Slope / Dist / Distlog）で確かめた（約290問）。
    #   Slope: eqangle, para, perp, constline だけ。Distlog: eqratio と cong。Dist: cong, rconst, coll。
    # cong は Dist と Distlog の両方で出る（保存した証明には AR としか記録がなく、区別できない）ので、
    # Length_chasing と Ratio_chasing のどちらでも書ける。expand は Length_chasing を使う。
    ("AR", frozenset({"eqangle", "para", "perp", "constline"}), "Angle_chasing"),
    ("AR", frozenset({"cong", "rconst", "coll"}), "Length_chasing"),
    ("AR", frozenset({"eqratio", "cong"}), "Ratio_chasing"),
    ("Transfer", None, "Transfer_between_equal_points"),
    ("r03", frozenset({"eqangle"}), "Arc_determines_internal_angles"),
    ("r04", frozenset({"cyclic"}), "Congruent_angles_are_in_a_circle"),
    ("r07", frozenset({"eqratio"}), "Thales_theorem_I"),
    ("r11", frozenset({"eqangle"}), "Bisector_theorem_I"),
    ("r12", frozenset({"eqratio"}), "Bisector_theorem_II"),
    ("r19", frozenset({"cong"}), "Hypotenuse_is_diameter"),
    ("r27", frozenset({"para"}), "Thales_theorem_II"),
    ("r28", frozenset({"coll"}), "Overlapping_parallels"),
    ("r34", frozenset({"simtri"}), "AA_similarity_of_triangles_direct"),
    ("r35", frozenset({"simtrir"}), "AA_similarity_of_triangles_reverse"),
    ("r41", frozenset({"para"}), "Thales_theorem_III"),
    ("r42", frozenset({"eqratio"}), "Thales_theorem_IV"),
    ("r43", frozenset({"perp"}), "Orthocenter_theorem"),
    ("r44", None, "Pappus_theorem"),
    ("r46", frozenset({"eqangle"}), "Incenter_theorem"),
    ("r49", frozenset({"cong"}), "Concyclic_point_is_equidistant_from_circumcenter"),
    ("r50", frozenset({"cong"}), "Center_of_cyclic_from_two_bisectors"),
    ("r51", frozenset({"rconst"}), "Midpoint_splits_in_two"),
    ("r52", frozenset({"eqratio", "eqangle"}), "Properties_of_similar_triangles_direct"),
    ("r53", frozenset({"eqratio", "eqangle"}), "Properties_of_similar_triangles_reverse"),
    ("r54", frozenset({"midp"}), "Definition_of_midpoint"),
    ("r56", frozenset({"coll", "cong"}), "Properties_of_midpoint"),
    ("r57", None, "Pythagoras_theorem"),
    ("r60", frozenset({"simtri"}), "SSS_similarity_of_triangles_direct"),
    ("r61", frozenset({"simtrir"}), "SSS_similarity_of_triangles_reverse"),
    ("r62", frozenset({"simtri"}), "SAS_similarity_of_triangles_direct"),
    ("r63", frozenset({"simtrir"}), "SAS_similarity_of_triangles_reverse"),
    ("r72", frozenset({"circle"}), "Equal_distances_define_circumcenter"),
    ("r73", None, "Circumcenter_has_equal_distances"),
    ("r82", frozenset({"para"}), "Collinear_implies_parallel"),
    ("r101", None, "Congruent_from_similar_triangles_direct"),
    ("r102", None, "Congruent_from_similar_triangles_reverse"),
    ("r103", frozenset({"cong", "eqangle"}), "Properties_of_congruent_triangles_direct"),
    ("r104", frozenset({"eqangle", "cong"}), "Properties_of_congruent_triangles_reverse"),
    ("r105", frozenset({"eqratio"}), "Ratio_of_collinear_points"),
    ("r106", None, "Definition_of_secant"),
    ("r107", frozenset({"eqpoint"}), "Two_distinct_lines_determine_a_unique_point"),
    ("r108", frozenset({"cong", "constline"}), "Coincident_points_have_equal_distances"),
    ("r109", frozenset({"eqpoint"}), "Collinear_points_with_equal_distances_coincide"),
    ("r110", frozenset({"cong"}), "Equal_angles_imply_equal_opposite_sides"),
    ("r110", frozenset({"contri"}), "SSS_congruence_of_triangles_direct"),
    ("r111", frozenset({"eqangle"}), "Equal_sides_imply_equal_opposite_angles"),
    ("r111", frozenset({"contrir"}), "SSS_congruence_of_triangles_reverse"),
    ("r112", frozenset({"contri"}), "SAS_congruence_of_triangles_direct"),
    ("r113", frozenset({"contrir"}), "SAS_congruence_of_triangles_reverse"),
]

_BY_NAME: dict[str, tuple[str, frozenset[str] | None]] = {}
for _code, _preds, _name in _ENTRIES:
    if _name in _BY_NAME:
        raise ValueError(f"duplicate rule name: {_name}")
    _BY_NAME[_name] = (_code, _preds)


def rule_name(code: str, pred: str) -> str:
    """GenesisGeo の規則の番号と結論の述語から、名前を返す。表にない組み合わせは ValueError。"""
    for c, preds, name in _ENTRIES:
        if c == code and (preds is None or pred in preds):
            return name
    raise ValueError(f"no name for rule {code!r} with conclusion {pred!r}")


def rule_code(name: str) -> str | None:
    """名前から GenesisGeo の規則の番号を返す。名前でなければ None。"""
    entry = _BY_NAME.get(name)
    return entry[0] if entry else None


def is_valid(name: str, pred: str) -> bool:
    """名前が表にあり、その規則が、この述語を結論にできるか。"""
    entry = _BY_NAME.get(name)
    return entry is not None and (entry[1] is None or pred in entry[1])


ALL_NAMES = tuple(_BY_NAME)
