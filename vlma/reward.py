"""報酬（ユーザーの方針）。値は仮で、決まっていない。

方針: 最短の導出過程は強い報酬、無駄のある導出過程は弱い報酬、証明に必要ない過程は負の報酬。
基準は、保存した証明（DDAR が最初に見つけた導出）。次のように手ごとの報酬を決める。

  導出全体の種類（trajectory_kind）
    shortest   目標に届き、却下された手も不要な手もなく、適用の手数が参照証明以下
    wasteful   目標に届くが、上に当てはまらない（不要な手や却下された手がある、手数が多い）
    incomplete 目標に届かない
  手ごとの報酬
    却下された手（書式・作図・結論・根拠の検証を通らない）   rejected
    受理された作図                                          0（作図の良し悪しは、導出の結果に反映される）
    受理された適用で、目標の導出に必要でない手             negative（真だが、証明に必要ない過程）
    受理された適用で、参照証明の手                         shortest なら strong、wasteful なら weak、incomplete なら 0
"""
from __future__ import annotations

from dataclasses import dataclass

from vlma.verify import Verdict


@dataclass(frozen=True)
class RewardConfig:
    strong: float = 1.0
    weak: float = 0.3
    negative: float = -0.3
    rejected: float = -1.0


def trajectory_kind(verdicts: list[Verdict], n_reference: int) -> str:
    if not any(v.goal_reached for v in verdicts):
        return "incomplete"
    n_apply = sum(v.ok and v.needed is not None for v in verdicts)
    bad = any(not v.ok or v.needed is False for v in verdicts)
    return "shortest" if not bad and n_apply <= n_reference else "wasteful"


def step_rewards(verdicts: list[Verdict], n_reference: int, cfg: RewardConfig = RewardConfig()) -> tuple[list[float], str]:
    kind = trajectory_kind(verdicts, n_reference)
    base = {"shortest": cfg.strong, "wasteful": cfg.weak, "incomplete": 0.0}[kind]
    out = []
    for v in verdicts:
        if not v.ok:
            out.append(cfg.rejected)
        elif v.needed is None:
            out.append(0.0)
        else:
            out.append(base if v.needed else cfg.negative)
    return out, kind
