import json
from pathlib import Path

import pytest
from PIL import Image

from vlma.figures import construction_texts, render_problem_figures
from vlma.steps import figure_path, problem_id, to_steps

RECORDS = json.loads((Path(__file__).parent / "data" / "records.json").read_text())


@pytest.mark.parametrize("i", range(len(RECORDS)))
def test_one_figure_per_construct_plus_initial(i, tmp_path):
    rec = RECORDS[i]
    figs = render_problem_figures(rec, tmp_path)
    assert len(figs) == len(construction_texts(rec)) + 1
    for rel, _ in figs:
        assert Image.open(tmp_path / rel).size == (512, 512)
    # 各ターンの image は、描いた図のどれかを指す
    paths = {rel for rel, _ in figs}
    assert {s["image"] for s in to_steps(rec)} == paths
    assert figure_path(problem_id(rec), 0) in paths
