# 規則名の対応表

手の出力（`APPLY <規則名> : ...`）に使う名前と、GenesisGeo の番号の対応。実装は `vlma/rules.py`。保存したデータ（`problems.jsonl`）は番号のまま持ち、名前に直すのは `expand` のとき。

**検査の内容:** 名前の一意性（コードとテストで検査）。名前と定義（前提 ⇒ 結論）の照合は、公式の規則ファイル（`rules.txt`）と C++ のソース（`DDAR/theorem.cpp`）を読んで行った。AR の種類は、C++ 版 DDAR が付ける理由で、約290問について確かめた。「データに出た手数」は、保存した10万問（680万手）での回数。

**公式の名前から変えたもの:** 定義と合わない、または意味が取りにくいため。r49, r50, r72, r73, r108, r109。r82 は、公式の名前が r28 と同じ（"Overlapping parallels"）で区別できないため。AR は、公式では番号が1つ（AR）で、結論の述語から3つに分けた。

| 名前 | 番号 | 結論にできる述語 | 内容（前提 ⇒ 結論） | データに出た手数 |
|---|---|---|---|---|
| Arc_determines_internal_angles | r03 | eqangle | cyclic A B P Q ⇒ eqangle P A P B Q A Q B | 279,679 |
| Congruent_angles_are_in_a_circle | r04 | cyclic | eqangle P A P B Q A Q B, ncoll P Q A B ⇒ cyclic A B P Q | 147,723 |
| Thales_theorem_I | r07 | eqratio | para A B C D, coll O A C, ncoll O A B, coll O B D ⇒ eqratio3 A B C D O O | 72,984 |
| Bisector_theorem_I | r11 | eqangle | eqratio d b d c a b a c, coll d b c, ncoll a b c ⇒ eqangle a b a d a d a c | 23,176 |
| Bisector_theorem_II | r12 | eqratio | eqangle a b a d a d a c, coll d b c, ncoll a b c ⇒ eqratio d b d c a b a c | 48,031 |
| Hypotenuse_is_diameter | r19 | cong | midp M A C, perp A B B C ⇒ cong A M B M | 52,428 |
| Thales_theorem_II | r27 | para | eqratio A O A C B O B D, coll O A C, coll O B D, ncoll A B C, sameside A O C B O D ⇒ para A B C D | 43,381 |
| Overlapping_parallels | r28 | coll | para A B A C ⇒ coll A B C | 214,137 |
| AA_similarity_of_triangles_direct | r34 | simtri | eqangle ×2, sameclock A B C P Q R ⇒ simtri A B C P Q R | 105,762 |
| AA_similarity_of_triangles_reverse | r35 | simtrir | eqangle ×2, sameclock A B C P R Q ⇒ simtrir A B C P Q R | 119,468 |
| Thales_theorem_III | r41 | para | eqratio m a m d n b n c, para a b c d, coll m a d, coll n b c, sameside ⇒ para m n a b | 4,428 |
| Thales_theorem_IV | r42 | eqratio | para a b c d, para m n a b, coll m a d, coll n b c, ncoll a b c ⇒ eqratio m a m d n b n c | 4,954 |
| Orthocenter_theorem | r43 | perp | perp a b c d, perp a c b d ⇒ perp a d b c | 7,975 |
| Pappus_theorem | r44 | （制限なし） | C++ 版で定義。データには出ていない | 0 |
| Incenter_theorem | r46 | eqangle | eqangle a b a x a x a c, eqangle b a b x b x b c, ncoll a b c ⇒ eqangle c b c x c x c a | 25,732 |
| Concyclic_point_is_equidistant_from_circumcenter | r49 | cong | circle O A B C, cyclic A B C D ⇒ cong O A O D （公式名: Recognize center of cyclic (circle)） | 15,033 |
| Center_of_cyclic_from_two_bisectors | r50 | cong | cong O A O B, cong O C O D, cyclic A B C D, npara A B C D ⇒ cong O A O C （公式名: Recognize center of cyclic (cong)） | 1,986 |
| Midpoint_splits_in_two | r51 | rconst | midp M A B ⇒ rconst M A A B 1/2 | 5 |
| Properties_of_similar_triangles_direct | r52 | eqratio, eqangle | simtri A B C P Q R ⇒ eqangle B A B C Q P Q R, eqratio B A B C Q P Q R | 330,418 |
| Properties_of_similar_triangles_reverse | r53 | eqratio, eqangle | simtrir A B C P Q R ⇒ eqangle B A B C Q R Q P, eqratio B A B C Q P Q R | 501,555 |
| Definition_of_midpoint | r54 | midp | cong M A M B, coll M A B ⇒ midp M A B | 14,859 |
| Properties_of_midpoint | r56 | coll, cong | midp M A B ⇒ coll M A B, cong M A M B | 90,199 |
| Pythagoras_theorem | r57 | （制限なし） | PythagoreanPremises a b c ⇒ PythagoreanConclusions a b c。規則ファイルにあるが、データには出ていない | 0 |
| SSS_similarity_of_triangles_direct | r60 | simtri | eqratio ×2, sameclock A B C P Q R ⇒ simtri A B C P Q R | 4,593 |
| SSS_similarity_of_triangles_reverse | r61 | simtrir | eqratio ×2, sameclock A B C P R Q ⇒ simtrir A B C P Q R | 9,563 |
| SAS_similarity_of_triangles_direct | r62 | simtri | eqratio, eqangle, sameclock A B C P Q R ⇒ simtri A B C P Q R | 158,125 |
| SAS_similarity_of_triangles_reverse | r63 | simtrir | eqratio, eqangle, sameclock A B C P R Q ⇒ simtrir A B C P Q R | 258,473 |
| Equal_distances_define_circumcenter | r72 | circle | cong O A O B, cong O A O C ⇒ circle O A B C （公式名: Congruence of circumcenter） | 12,789 |
| Circumcenter_has_equal_distances | r73 | （制限なし） | circle O A B C ⇒ cong O A O B, cong O A O C, cong O B O C （公式名: Circumcenter of congruent triangles）。データには出ていない | 0 |
| Collinear_implies_parallel | r82 | para | coll A B C ⇒ para A B B C, para A B A C （公式名は r28 と同じ Overlapping parallels） | 517,831 |
| Congruent_from_similar_triangles_direct | r101 | （制限なし） | simtri A B C P Q R, cong A B P Q ⇒ contri A B C P Q R。データには出ていない | 0 |
| Congruent_from_similar_triangles_reverse | r102 | （制限なし） | simtrir …, cong A B P Q ⇒ contrir …。データには出ていない | 0 |
| Properties_of_congruent_triangles_direct | r103 | cong, eqangle | contri ⇒ cong ×3, eqangle ×3 | 27,128 |
| Properties_of_congruent_triangles_reverse | r104 | eqangle, cong | contrir ⇒ cong ×3, eqangle ×3 | 193,171 |
| Ratio_of_collinear_points | r105 | eqratio | eqratio A B : A C = P Q : P R, coll, coll ⇒ eqratio A B : B C = P Q : Q R ほか | 72,615 |
| Definition_of_secant | r106 | （制限なし） | C++ 版で定義。データには出ていない | 0 |
| Two_distinct_lines_determine_a_unique_point | r107 | eqpoint | p, q が直線 a b と直線 c d の両方の上にあり、ncoll a b d ⇒ eqpoint p q | 1,207 |
| Coincident_points_have_equal_distances | r108 | cong, constline | eqpoint A B ⇒ cong P A P B, constline P A B （公式名: Distances to overlapped point are equal） | 820 |
| Collinear_points_with_equal_distances_coincide | r109 | eqpoint | coll A P1 C, coll A P2 C, cong A P1 A P2, cong C P1 C P2 ⇒ eqpoint P1 P2 （公式名: Same ratio on same line implies equal points。実際の前提は「比」ではなく「等しい距離」） | 1,758 |
| Equal_angles_imply_equal_opposite_sides | r110 | cong | eqangle（頂点 a b の底角が等しい）⇒ cong 頂点 a, 頂点 b | 158,417 |
| SSS_congruence_of_triangles_direct | r110 | contri | cong ×3, sameclock ⇒ contri | 5,360 |
| Equal_sides_imply_equal_opposite_angles | r111 | eqangle | cong 頂点 a, 頂点 b ⇒ eqangle（底角が等しい） | 413,569 |
| SSS_congruence_of_triangles_reverse | r111 | contrir | cong ×3, sameclock ⇒ contrir | 94,252 |
| SAS_congruence_of_triangles_direct | r112 | contri | cong ×2, eqangle, sameclock ⇒ contri | 21,329 |
| SAS_congruence_of_triangles_reverse | r113 | contrir | cong ×2, eqangle, sameclock ⇒ contrir | 75,266 |
| Angle_chasing | AR | eqangle, para, perp, constline | 角度（傾き）の代数的推論。C++ 版の理由は "AR with Slope" | 1,605,071 |
| Length_chasing | AR | cong, rconst, coll | 長さの代数的推論。"AR with Dist"。cong は Distlog のものも含めて、展開のときはこの名前にする | 470,815 |
| Ratio_chasing | AR | eqratio, cong | 長さの比の代数的推論。"AR with Distlog"。展開のときは eqratio だけがこの名前になる | 593,190 |
| Transfer_between_equal_points | Transfer | （制限なし） | 同じ位置の2点の間で、事実を移す。C++ 版の理由は "Transfer" | 4,175 |

**注意:**
- r110 と r111 は、結論の述語によって別の定理を指す（番号が一意でない）。名前では区別される。
- AR のうち cong は、C++ 版では Dist と Distlog の両方で出る（約3,100対1,200）。保存した証明には AR としか記録がなく、区別できないので、Length_chasing と Ratio_chasing のどちらでも書ける。データを展開するときは Length_chasing にする。
- 手数は、展開で付く名前での回数（Length_chasing は cong 436,750 + coll 33,621 + rconst 444）。AR の合計は 2,669,076 で、Angle_chasing + Length_chasing + Ratio_chasing と一致する。
- 名前は検証器で確かめる: 表にない名前や、その規則が結論にできない述語の手は、`unknown_rule` で却下する。規則名がその手に実際に使えるか（その規則だけで結論が導けるか）は、まだ検証していない。
