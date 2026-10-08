"""「情報」欄: 直前までに適用した手に関わる点の、長さと角度の数値。

図の座標から機械的に計算するので、証明に依存しない。適用した手の結論に出てくる点（作図なら新しい点と、その条件に出てくる点）について、
その点どうしの長さと、3点の角度を、手を適用したあとに追記する。例えば三平方の定理を三角形 abc に適用すると、
その三辺と三つの角が追記される。同じ長さ・角度は繰り返さない。
"""
from __future__ import annotations

import math
from itertools import combinations

MAX_ANGLE_POINTS = 4  # 点が5個以上の手（eqangle の8点など）では、長さだけにして長くなりすぎるのを防ぐ


class InfoLog:
    """手ごとに追記される数値を持つ。text() が、その時点の情報欄になる。"""

    def __init__(self, digits: int = 4):
        self._f = f"{{:.{digits}f}}"
        self._lengths: dict[frozenset, str] = {}
        self._angles: dict[tuple, str] = {}

    def add(self, points: dict[str, list[float]], names: list[str]) -> None:
        names = [n for n in dict.fromkeys(names) if n in points]
        for a, b in combinations(names, 2):
            key = frozenset((a, b))
            if key not in self._lengths:
                x, y = sorted((a, b))
                self._lengths[key] = f"{x}{y} {self._f.format(math.dist(points[a], points[b]))} ;"
        if len(names) > MAX_ANGLE_POINTS:
            return
        for b in names:
            for a, c in combinations([n for n in names if n != b], 2):
                key = (b, frozenset((a, c)))
                if key in self._angles:
                    continue
                v1 = (points[a][0] - points[b][0], points[a][1] - points[b][1])
                v2 = (points[c][0] - points[b][0], points[c][1] - points[b][1])
                n1, n2 = math.hypot(*v1), math.hypot(*v2)
                if n1 == 0 or n2 == 0:
                    continue
                cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
                x, z = sorted((a, c))
                self._angles[key] = f"{x}{b}{z} {math.degrees(math.acos(cos)):.1f} ;"

    def text(self) -> str:
        parts = []
        if self._lengths:
            parts.append("<lengths> " + " ".join(self._lengths.values()) + " </lengths>")
        if self._angles:
            parts.append("<angles> " + " ".join(self._angles.values()) + " </angles>")
        return " ".join(parts)
