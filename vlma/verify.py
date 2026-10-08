"""モデルの出力を DDAR で検証する（R2-3）。

手の検証は2段階で行う。
  1. 結論が真か: 問題と、ここまでに受理した作図に DDAR を飽和まで実行し、結論が成立するか（偽の主張を除く）。
  2. 書いた根拠から導けるか: 根拠の番号を、述べた事実に直し、その事実だけを前提に DDAR を1段だけ実行して結論が導けるか。
     これがないと、作図のあとに目標の文をそのまま出せば、1手で完了になる。さらに、書いた規則だけを有効にした DDAR でも
     導けることを確認する（規則名がその手に使えるか。AR なら代数的推論だけ）。通らなければ rule_mismatch。1段にするのは、複数段の推論を1手に押し込む近道を防ぐため
     （circle だけは2段。DERIVATION_LEVEL を参照）。保存した証明の850問（36,843手）は、この条件ですべて通る。
DDAR は生成機・公式の探索と同じ C++ 版を使う（Python 版は退化した配置などで導出が弱く、正しい手を落とす）。
規則名や根拠の番号が正しいかは見ない。完了は、受理した手の結論が目標と一致したときとする。
（飽和状態は目標を最初から含むので、「飽和状態で目標が成立」を完了の条件にはできない。）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from newclid.api import CSolver, GeometricSolverBuilder
import re

from newclid.configs import load_solver_config
from newclid.DDAR.build import DDAR
from newclid.discovery.extraction import parsing
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions
from newclid.statement import Statement

from vlma import actions as A
from vlma import rules
from vlma.steps import problem_points


# 書いた根拠から導けるかの確認で、DDAR を実行する段数。既定は1段（複数段の推論を1手に押し込む近道を防ぐため）。
# circle だけは2段にする。1段では、根拠が揃っていても導けない手がある（保存した証明の199手のうち5手）。2段なら199手すべてで導ける。
# 他の述語は、保存した証明の12,504手のうち、1段ですべて導けた（丸めていない座標で測定）。
DERIVATION_LEVEL = {"circle": 2}


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str | None = None  # 却下の理由（A.REJECT_REASONS のどれか）。受理なら None
    goal_reached: bool = False
    detail: str = ""
    # 受理した apply の手について、保存した証明（reference）の導出に必要な手か。
    # True: 参照証明の手。False: 真だが目標に必要ない手、または既に述べた事実の言い直し。None: 判定しない（作図、reference なし）
    needed: bool | None = None


class StepVerifier:
    """1問ぶんの状態を持ち、操作を1つずつ受け取って受理・却下を返す。"""

    def __init__(self, record: dict[str, Any], *, seed: int = 123, timeout: int = 120, engine: str = "both",
                 reference: list[str] | None = None):
        premises, goal = record["fl_problem"].split("?", 1)
        self._premises = premises.strip().rstrip(";")
        self._goal_text = goal.strip()
        self._llm_goal = parsing.extract_goal(record["llm_input_renamed"])
        self._seed = seed
        self._timeout = timeout
        # DDAR の実装は2つあり、導出できる事実が少し違う。
        #  "c": 生成機・公式の探索と同じ C++ 版。代数的推論（AR）で導いた角度などが、Python 側の状態には戻らない。
        #  "python": Python 版。退化した配置（3点が一直線で para になる場合など）で導出が弱い。
        #  "both": どちらかで成立すれば受理する（既定）。2つめは、1つめで成立しなかったときだけ実行する。
        if engine not in ("c", "python", "both"):
            raise ValueError(engine)
        self._engines = ("c", "python") if engine == "both" else (engine,)
        self._aux: list[str] = []  # 受理した作図（GenesisGeo の作図言語）
        self._points = set(problem_points(record["llm_input_renamed"]))
        self._reference = [a for a in (A.parse_action(r) for r in reference or []) if isinstance(a, A.Action) and a.kind == "apply"]
        self._has_reference = reference is not None
        self._ref_keys: set[str] | None = None  # 参照証明の各手の結論（正規化した文字列。DDAR の実装によらず比べられる）
        self._stated_keys: set[str] = set()  # 述べた手の結論（言い直しの判定に使う）
        self._facts: dict[str, tuple[str, list[str]]] = {}  # 番号 → 述べた事実（述語, 引数）
        body = parsing.strip_tags(record["llm_input_renamed"]).split("?", 1)[0]
        for pred, args, fid in A.parse_facts(body):
            self._facts[fid] = (pred, list(args))
        self._config = None
        self._saturated: dict[str, Any] = {}  # 実装ごとの、現在の作図での飽和済みの状態（作図が変わると捨てる）

    @property
    def constructions(self) -> list[str]:
        """ここまでに受理した作図（GenesisGeo の作図言語）。受理した作図を含む図を描くのに使う。"""
        return list(self._aux)

    def _problem_text(self, extra: list[str], with_goal: bool) -> str:
        cons = "; ".join([self._premises, *self._aux, *extra])
        return f"{cons} ? {self._goal_text}" if with_goal else cons

    def _saturated_proof(self, engine: str):
        if engine not in self._saturated:
            builder = GeometricSolverBuilder(seed=self._seed).load_problem_from_txt(
                self._problem_text([], with_goal=True)
            ).del_goals()
            solver = builder.build()
            if engine == "c":
                CSolver(solver=solver).run(max_level=500)  # 導出した事実を solver.proof に書き戻す
            else:
                solver.run(timeout=self._timeout)
            self._saturated[engine] = solver.proof
        return self._saturated[engine]

    def check(self, text: str) -> Verdict:
        action = A.parse_action(text)
        if isinstance(action, A.ParseFailure):
            return Verdict(False, A.PARSE_FAILED)
        if action.kind == "construct":
            return self._check_construct(action)
        return self._check_apply(action)

    def _check_construct(self, action: A.Action) -> Verdict:
        if action.point in self._points:
            return Verdict(False, A.BUILD_FAILED, detail="point already exists")
        cons = try_full_aux_dsl_to_constructions(action.to_genesis())
        if cons is None:
            return Verdict(False, A.BUILD_FAILED, detail="construction not translatable")
        try:
            GeometricSolverBuilder(seed=self._seed).load_problem_from_txt(
                self._problem_text([cons], with_goal=True)
            ).build()
        except Exception as exc:  # noqa: BLE001  図が作れない原因は多様なので理由だけ残す
            return Verdict(False, A.BUILD_FAILED, detail=f"{type(exc).__name__}: {exc}"[:200])
        self._aux.append(cons)
        for pred, args, fid in A.parse_facts(action.body):
            self._facts[fid] = (pred, list(args))
        self._points.add(action.point)
        self._saturated = {}
        return Verdict(True)

    def _check_apply(self, action: A.Action) -> Verdict:
        if not rules.is_valid(action.rule, action.pred):
            return Verdict(False, A.UNKNOWN_RULE, detail=f"rule {action.rule!r} cannot conclude {action.pred!r}")
        if not set(action.point_args) <= self._points:
            return Verdict(False, A.INVALID_STATEMENT, detail="unknown point")
        solver_errors: list[str] = []
        for engine in self._engines:
            try:
                proof = self._saturated_proof(engine)
            except Exception as exc:  # noqa: BLE001  退化した図（点が重なるなど）で、DDAR の実装が例外を出すことがある。そのエンジンは使わない
                solver_errors.append(f"{engine}: {type(exc).__name__}: {exc}"[:120])
                continue
            dg = proof.dep_graph
            try:
                st = Statement.from_tokens(action.conclusion_tokens, dg)
            except Exception as exc:  # noqa: BLE001  未知の述語や引数の数の違い
                return Verdict(False, A.INVALID_STATEMENT, detail=f"{type(exc).__name__}: {exc}"[:200])
            if st is None:
                return Verdict(False, A.INVALID_STATEMENT, detail="statement not constructible")
            if st.check():
                break
        else:
            return Verdict(False, A.NOT_HOLD, detail="; ".join(solver_errors))
        # 2. 書いた根拠の事実だけから導けるか
        if any(p not in self._facts for p in action.premises):
            return Verdict(False, A.INVALID_CITATION, detail="cited fact is not stated")
        if not self._derivable_from_cited(action, None):
            return Verdict(False, A.NOT_DERIVABLE)
        if not self._rule_fits(action):
            return Verdict(False, A.RULE_MISMATCH, detail=f"{action.rule} alone cannot derive the conclusion from the cited facts")
        key = st.predicate.to_str(st)
        redundant = key in self._stated_keys
        self._stated_keys.add(key)
        self._facts[action.fact_id] = (action.pred, list(action.args))
        needed = None
        if self._has_reference:
            if self._ref_keys is None:
                refs = (Statement.from_tokens(r.conclusion_tokens, dg) for r in self._reference)
                self._ref_keys = {r.predicate.to_str(r) for r in refs if r is not None}
            needed = key in self._ref_keys and not redundant
        reached = False
        if self._llm_goal is not None:
            goal_st = Statement.from_tokens((self._llm_goal.predicate,) + tuple(self._llm_goal.args), dg)
            reached = goal_st is not None and st == goal_st
        return Verdict(True, goal_reached=reached, needed=needed)

    def _rule_fits(self, action: A.Action) -> bool:
        """書いた規則が、その手に使えるか。

        Transfer は定理ではなく、同じ位置の2点（eqpoint）の間で事実を置き換えるエンジンの仕組みで、特定の規則だけを有効にしても
        再現できない。そこで、根拠に eqpoint の事実を含むことだけを確認する（保存した証明の4,175手すべてが満たす）。
        AR（代数的推論）は、AR だけを有効にした DDAR で導けるか。
        それ以外は、書いた規則だけを有効にした DDAR で、書いた根拠から結論が導けるか。規則が直接導く形を、AR で並べ替えた結論
        （例: Thales_theorem_I が導く eqratio3 を、別の eqratio の並びにしたもの）は、規則だけでは導けないので、
        次の場合も認める: 規則と AR を併用すれば導け、かつ AR だけでは導けない（つまり、その規則が必要）。
        """
        code = rules.rule_code(action.rule)
        if code == "Transfer":
            return any(self._facts[p][0] == "eqpoint" for p in action.premises)
        if code == "AR":
            return self._derivable_from_cited(action, self._rule_only_config("AR"))
        if self._derivable_from_cited(action, self._rule_only_config(code)):
            return True
        with_ar = self._rule_only_config(code)
        with_ar["using_ar"] = True
        return (self._derivable_from_cited(action, with_ar)
                and not self._derivable_from_cited(action, self._rule_only_config("AR")))

    def _rule_only_config(self, code: str) -> dict:
        """その規則だけを有効にした設定。AR は代数的推論だけ。"""
        if self._config is None:
            self._config = load_solver_config(None)
        cfg = dict(self._config)
        for k in list(cfg):
            if re.fullmatch(r"r\d+", k):
                cfg[k] = False
        cfg["using_ar"] = code == "AR"
        if re.fullmatch(r"r\d+", code):
            cfg[code] = True
        return cfg

    def _derivable_from_cited(self, action: A.Action, config: dict | None) -> bool:
        """書いた根拠の事実だけを前提に、C++ 版 DDAR を1段実行して、結論が導けるか。config が None なら全規則を有効にする。"""
        if self._config is None:
            self._config = load_solver_config(None)
        proof = next(iter(self._saturated.values()))
        pts = [(n, p.num.x, p.num.y) for n, p in proof.symbols_graph.name2node.items() if hasattr(p.num, "x")]
        premises = [self._facts[p] for p in action.premises]
        level = DERIVATION_LEVEL.get(action.pred, 1)
        solved, _ = DDAR.run_ddar("x", pts, premises, [(action.pred, list(action.args))], level,
                                  self._config if config is None else config)
        return bool(solved)
