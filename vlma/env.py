"""ベンチマークの環境（1問 = 1エピソード）。モデルの出力を1手ずつ受け取り、検証して、次の状態を返す。

状態は、学習データ（vlma.steps.expand）の1ターンの入力と同じもの: 問題文、情報欄、過去の手、図。
受理した手は、書式をそろえて（action.format()）過去の手に足す。作図を受理したら、その作図を含む図を描き直し、
その図の座標で情報欄に数値を足す（学習データと同じ手順）。却下した手は、状態を変えない。
検証と図の置き場所（seed）は、問題ごとの placement_seed を使う（学習データの作成時と同じ）。
newclid が要るので、genesisgeo のコンテナで動かす。

モード（mode）:
  full       モデルが作図も推論の手も全部書く。1手ずつ検証し、目標の手に届いたら成功。
  construct  AlphaGeometry / GenesisGeo と同じ形。モデルは補助点の作図だけを出し、作図を受理するたびに記号エンジン（DDAR）で
             目標が証明できるかを見る。できたら成功。モデルが APPLY を出したら「作図は終わり」として終了する
             （学習データは、作図が先、推論が後の順）。作図なしで DDAR だけで解ける問題は、リセット時に成功にして、
             solved_alone に記録する（モデルの手柄にしない）。
参照証明（compact["actions"]）が無い問題（IMO-AG-30 など）も扱う。そのとき n_reference は None で、試行の上限は固定値。
"""
from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any

from newclid.api import GeometricSolverBuilder
from newclid.evaluation.search_runtime import run_ddar_c

from vlma import actions as A
from vlma.figures import figure_points, problem_text, render_figure
from vlma.info import InfoLog
from vlma.reward import RewardConfig, step_rewards
from vlma.steps import _action_points, expand, figure_path
from vlma.verify import StepVerifier, Verdict

_RENDER_LOCK = threading.Lock()  # matplotlib の pyplot は、スレッドをまたいで使えない
NO_REFERENCE_ATTEMPTS = 150  # 参照証明が無い問題の、full モードの試行の上限（既定）
MODES = ("full", "construct")


class Episode:
    def __init__(self, compact: dict[str, Any], data_dir: str | Path, out_dir: str | Path, max_attempts: int | None = None,
                 mode: str = "full", max_constructs: int = 4):
        if mode not in MODES:
            raise ValueError(mode)
        self.c = compact
        self.pid = compact["problem_id"]
        self.data_dir, self.out_dir = Path(data_dir), Path(out_dir)
        self.mode, self.max_constructs = mode, max_constructs
        if compact["actions"]:
            self.reference = [s["output"] for s in expand(compact)]
            self.n_reference: int | None = sum(not o.startswith("CONSTRUCT") for o in self.reference)  # 参照証明の、適用の手数
        else:  # 参照証明が無い問題
            self.reference, self.n_reference = [], None
        self.record = {"fl_problem": compact["fl_problem"], "llm_input_renamed": compact["problem"]}
        self.seed = compact["placement_seed"]
        self.verifier = StepVerifier(self.record, seed=self.seed, reference=self.reference or None)
        n = len(self.reference)
        if max_attempts:
            self.max_attempts = max_attempts
        elif mode == "construct":
            self.max_attempts = 4 * max_constructs  # 却下を含めた作図の試行
        else:
            self.max_attempts = min(400, max(2 * n, n + 20)) if n else NO_REFERENCE_ATTEMPTS
        self.info = InfoLog()
        self.history: list[str] = []
        self.figure_index = 0
        self.points = compact["figure_points"][0]
        self.image = self._initial_image()
        self.show_image = True  # 最初のターンと、作図の手のあと
        self.outputs: list[str] = []  # 試みた出力（却下を含む）
        self.verdicts: list[Verdict] = []
        self.done = False
        self.stopped = False  # construct: モデルが APPLY を出して、作図を終えた
        self.proved = False  # construct: 作図のあとに DDAR が目標を証明した
        self.solved_alone: bool | None = None
        if mode == "construct":
            self.solved_alone = self.proved = self.done = self._proves_goal()  # 作図なし（DDAR だけ）で解けるなら、そこで終わり
        self.lock = threading.Lock()

    def _initial_image(self) -> str:
        path = self.data_dir / figure_path(self.pid, 0)
        if not path.exists():  # 学習データの図が無いとき（試験用）は、描く
            path = self.out_dir / figure_path(self.pid, 0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _RENDER_LOCK:
                render_figure(problem_text(self.record, []), path, self.seed)
        return str(path)

    def _proves_goal(self) -> bool:
        """ここまでに受理した作図を加えた問題で、C++ 版 DDAR（GenesisGeo の評価と同じ run_ddar_c）が目標を証明できるか。"""
        try:
            text = problem_text(self.record, self.verifier.constructions)
            proof = GeometricSolverBuilder(seed=self.seed).load_problem_from_txt(text).build(max_attempts=100).proof
            return bool(run_ddar_c(proof))
        except Exception:  # noqa: BLE001  退化した図などで、構築や DDAR が例外を出したら、証明できないとする
            return False

    def state(self) -> dict[str, Any]:
        return {"problem": self.c["problem"], "info": self.info.text(), "history": list(self.history),
                "image": self.image if self.show_image else None, "attempts": len(self.outputs), "accepted": len(self.history),
                "max_attempts": self.max_attempts, "done": self.done}

    def step(self, text: str) -> dict[str, Any]:
        """出力を1つ検証する。受理なら状態を進める。"""
        with self.lock:
            if self.done:
                raise RuntimeError("episode is already finished")
            if self.mode == "construct":
                parsed = A.parse_action(text)
                if isinstance(parsed, A.Action) and parsed.kind == "apply":  # 作図は終わり（検証しない）
                    self.outputs.append(text)
                    self.stopped = self.done = True
                    return {"verdict": {"ok": False, "reason": "stop_constructing", "goal_reached": False, "needed": None, "detail": ""},
                            "done": True, "state": self.state()}
            try:
                verdict = self.verifier.check(text)
            except Exception as exc:  # noqa: BLE001  検証器の例外（想定外の図など）で、環境を落とさない。却下として記録し、理由を残す
                verdict = Verdict(False, A.NOT_HOLD, detail=f"verifier error: {type(exc).__name__}: {exc}"[:200])
            self.outputs.append(text)
            self.verdicts.append(verdict)
            if verdict.ok:
                action = A.parse_action(text)
                self.history.append(action.format())
                if action.kind == "construct":
                    self.figure_index += 1
                    path = self.out_dir / figure_path(self.pid, self.figure_index)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with _RENDER_LOCK:
                        proof = render_figure(problem_text(self.record, self.verifier.constructions), path, self.seed)
                    self.points = figure_points(proof)
                    self.image = str(path)
                self.info.add(self.points, _action_points(action))
                self.show_image = action.kind == "construct"
                if self.mode == "construct":
                    self.proved = self._proves_goal()
            if self.mode == "construct":
                n_cons = sum(h.startswith("CONSTRUCT") for h in self.history)
                self.done = self.proved or n_cons >= self.max_constructs or len(self.outputs) >= self.max_attempts
            else:
                self.done = verdict.goal_reached or len(self.outputs) >= self.max_attempts
            return {"verdict": {"ok": verdict.ok, "reason": verdict.reason, "goal_reached": verdict.goal_reached or self.proved,
                                "needed": verdict.needed, "detail": verdict.detail},
                    "done": self.done, "state": self.state()}

    def result(self, cfg: RewardConfig = RewardConfig()) -> dict[str, Any]:
        """エピソードの結果。目標に届いたか、導出の種類（shortest / wasteful / incomplete）、手ごとの報酬の合計など。

        参照証明が無い問題と、construct モードでは、導出の種類と報酬は出さない（kind は solved / unsolved）。
        """
        success = self.proved if self.mode == "construct" else any(v.goal_reached for v in self.verdicts)
        if self.n_reference is None or self.mode == "construct":
            rewards, kind = [], "solved" if success else "unsolved"
        else:
            rewards, kind = step_rewards(self.verdicts, self.n_reference, cfg)
        rejected = [(k, v.reason) for k, v in enumerate(self.verdicts) if not v.ok]
        return {
            "problem_id": self.pid, "mode": self.mode, "success": success, "kind": kind,
            "constructs": sum(h.startswith("CONSTRUCT") for h in self.history), "solved_alone": self.solved_alone, "stopped": self.stopped,
            "n_reference": self.n_reference, "n_reference_turns": len(self.reference),
            "attempts": len(self.outputs), "accepted": len(self.history), "rejected": len(rejected),
            "reject_reasons": dict(Counter(r for _, r in rejected)), "first_reject_attempt": rejected[0][0] if rejected else None,
            "needed": sum(v.ok and v.needed is True for v in self.verdicts),
            "unneeded": sum(v.ok and v.needed is False for v in self.verdicts),
            "reward_total": round(sum(rewards), 4), "outputs": self.outputs, "reasons": [v.reason for v in self.verdicts],
            "max_attempts": self.max_attempts,
        }
