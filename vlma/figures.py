"""作図ごとの図を描く（R2-4）。

GenesisGeo の推論時の描画（agent/vllm.py の _build_vl_request_payload）と同じ関数を使う。
学習時と推論時で図の見た目をそろえるため、生成機が描いた画像（image_path）は使わず、ここで描き直す。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from newclid.api import GeometricSolverBuilder  # noqa: E402
from newclid.discovery.extraction import parsing  # noqa: E402
from newclid.evaluation.search_runtime import try_full_aux_dsl_to_constructions  # noqa: E402
from newclid.generation.writer import save_figure_as_png  # noqa: E402
from newclid.numerical.draw_clause_figure import draw_clause_figure  # noqa: E402
from newclid.statement import Statement  # noqa: E402

from vlma.actions import Action  # noqa: E402
from vlma.steps import extract_actions, figure_path, problem_id  # noqa: E402

IMG_PIXELS = 512  # GenesisGeo の推論時と同じ


def construction_texts(record: dict[str, Any]) -> list[str]:
    """作図の手を、GenesisGeo の作図言語（`e = perp ..., cong ...`）に1手ずつ変換する。"""
    out = []
    for a in extract_actions(record):
        if isinstance(a, Action) and a.kind == "construct":
            cons = try_full_aux_dsl_to_constructions(a.to_genesis())
            if cons is None:
                raise ValueError(f"construction not translatable: {a.format()}")
            out.append(cons)
    return out


def problem_text(record: dict[str, Any], constructions: list[str]) -> str:
    premises, goal = record["fl_problem"].split("?", 1)
    return "; ".join([premises.strip().rstrip(";"), *constructions]) + f" ? {goal.strip()}"


def render_figure(text: str, out_png: str | Path, seed: int = 0):
    """問題文（作図を含む）から図を描いて保存し、数値の状態（ProofState）を返す。

    補助点の置き場所は seed で決まる（円と円の交点など、2通りある作図がある）。
    検証器（vlma.verify）と同じ seed を使うこと。
    """
    builder = GeometricSolverBuilder(seed=seed).load_problem_from_txt(text)
    proof = builder.build().proof
    problem = builder.problemJGEX
    fig = draw_clause_figure(proof, problem, None, proof.rng, draw_annotations=True, theme=None)
    save_figure_as_png(fig, png_path=str(out_png), img_pixels=IMG_PIXELS, direct_png=True)
    plt.close(fig)
    return proof


def figure_points(proof) -> dict[str, list[float]]:
    """図の各点の座標（小数6桁）。情報欄（vlma.info）の元データとして保存する。"""
    return {n: [round(p.num.x, 6), round(p.num.y, 6)]
            for n, p in proof.symbols_graph.name2node.items() if hasattr(p.num, "x")}


def recorded_facts(record: dict[str, Any]) -> list[tuple[str, ...]]:
    """記録された、図で数値的に成り立つべき事実（numerical_check、trivial、各手の結論）。"""
    out = record["llm_output_renamed"]
    facts = []
    for tag in ("numerical_check", "trivial"):
        for pred, args, _ in parsing.parse_fact_segments(parsing.extract_tag_content(out, tag)):
            facts.append((pred, *args))
    for a in extract_actions(record):
        if isinstance(a, Action) and a.kind == "apply":
            facts.append(a.conclusion_tokens)
    return facts


def find_placement_seed(record: dict[str, Any], max_seed: int = 64) -> int | None:
    """記録された事実がすべて数値的に成り立つ、補助点の置き場所（seed）を探す。見つからなければ None。

    生成機は補助点の座標を記録しないため、2通りある作図では、向き（sameclock など）が合う方を探す。
    """
    text = problem_text(record, construction_texts(record))
    facts = recorded_facts(record)
    build_failures = 0
    for seed in range(max_seed):
        try:
            # max_attempts は公式の推論時（build_problem_proof）と同じ 100。既定の10000では、置けない作図で非常に遅い
            proof = GeometricSolverBuilder(seed=seed).load_problem_from_txt(text).build(max_attempts=100).proof
            dg = proof.dep_graph
            if all((st := Statement.from_tokens(t, dg)) is not None and st.check_numerical() for t in facts):
                return seed
        except Exception:  # noqa: BLE001  この seed では作図が置けない（点が近すぎるなど）
            build_failures += 1
            if build_failures >= 3:  # 作図そのものが置けない問題は、全 seed を試さず諦める
                return None
    return None


def render_problem_figures(record: dict[str, Any], out_root: str | Path, seed: int = 0) -> list[tuple[str, Any]]:
    """作図の数 + 1 枚の図を描く。figure_index k の図は、最初の k 個の作図を含む。"""
    pid = problem_id(record)
    cons = construction_texts(record)
    out = []
    for k in range(len(cons) + 1):
        rel = figure_path(pid, k)
        path = Path(out_root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        proof = render_figure(problem_text(record, cons[:k]), path, seed)
        out.append((rel, proof))
    return out
