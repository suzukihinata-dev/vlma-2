import json
import random
from pathlib import Path

from vlma import data
from vlma.steps import expand, to_compact

RECORDS = json.loads((Path(__file__).parent / "data" / "records.json").read_text())
PTS = {"a": [0.0, 0.0], "b": [3.0, 0.0], "c": [0.0, 4.0], "d": [1.0, 1.0], "e": [2.0, 2.0], "f": [1.0, 3.0], "g": [2.0, 1.0]}


def make_dir(tmp_path, n=3):
    compacts = []
    for rec in RECORDS[:n]:
        c = to_compact(rec)
        n_fig = sum(a.startswith("CONSTRUCT") for a in c["actions"]) + 1
        c["figure_points"] = [PTS] * n_fig
        c["n_turns"] = len(c["actions"])
        compacts.append(c)
    with open(tmp_path / "problems.jsonl", "w") as f:
        for c in compacts:
            f.write(json.dumps(c) + "\n")
    split = {c["problem_id"]: ("train" if i < n - 1 else "val") for i, c in enumerate(compacts)}
    json.dump(split, open(tmp_path / "split.json", "w"))
    return compacts


def test_image_is_shown_only_when_the_figure_changed():
    steps = expand({**to_compact(RECORDS[1]), "figure_points": [PTS] * 3})
    shown = [data.shows_image(steps, k) for k in range(len(steps))]
    assert shown[0] and sum(shown) == 3  # 最初のターンと、2つの作図のあと
    for k, s in enumerate(steps):
        if k > 0 and steps[k - 1]["output"].startswith("CONSTRUCT"):
            assert shown[k]
    assert not shown[-1]


def test_prompt_layout_and_loss_part():
    steps = expand({**to_compact(RECORDS[0]), "figure_points": [PTS, PTS]})
    prompt, target = data.build_text(steps[3], with_image=False)
    assert prompt.startswith("<problem>") and "<history>" in prompt and "<info>" in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")  # thinking の枠（中身は空）
    assert target == steps[3]["output"] + data.END
    first, _ = data.build_text(steps[0], with_image=True)
    assert first.startswith(data.IMAGE_MARK) and "<history>" not in first and "<info>" not in first  # 最初は何も追記されていない
    # 履歴は、前の手を1行に1つずつ
    hist = prompt.split("<history>\n")[1].split("\n</history>")[0].split("\n")
    assert hist == steps[3]["history"]


def test_choose_turns_keeps_all_constructs_and_is_deterministic():
    construct = [0, 7]
    a = data.choose_turns(30, construct, 4, random.Random("x"))
    b = data.choose_turns(30, construct, 4, random.Random("x"))
    assert a == b and len(a) == 4 and {0, 7} <= set(a) and a == sorted(a)
    assert data.choose_turns(30, [0, 1, 2, 3, 4], 3, random.Random(0)) == [0, 1, 2, 3, 4]  # 作図が多ければ、作図だけ


def test_all_turns_are_used_by_default_and_the_problem_count_is_what_limits_the_amount(tmp_path):
    compacts = make_dir(tmp_path)
    full = data.VlmaDataset(tmp_path, "train", None)  # turns_per_problem を指定しない: 各問題の全ターン
    assert len(full) == sum(len(c["actions"]) for c in compacts[:2])
    assert data.choose_turns(30, [0, 7], None, random.Random(0)) == list(range(30))
    one = data.VlmaDataset(tmp_path, "train", None, max_problems=1)  # 量は、問題数で絞る
    assert len(one) == len(compacts[0]["actions"])
    # 問題の一覧（subset）を渡すと、その問題だけ、その順番で使う
    json.dump([compacts[1]["problem_id"]], open(tmp_path / "subset.json", "w"))
    sub = data.VlmaDataset(tmp_path, "train", None, subset=tmp_path / "subset.json")
    assert [p[0] for p in sub.problems] == [compacts[1]["problem_id"]] and len(sub) == len(compacts[1]["actions"])


def test_dataset_selects_by_split_and_epoch(tmp_path):
    compacts = make_dir(tmp_path)
    tr = data.VlmaDataset(tmp_path, "train", None, turns_per_problem=4, seed=1)
    va = data.VlmaDataset(tmp_path, "val", None, turns_per_problem=4, seed=1)
    assert {tr.problems[i][0] for i, _ in tr.index} == {c["problem_id"] for c in compacts[:2]}
    assert {va.problems[i][0] for i, _ in va.index} == {compacts[2]["problem_id"]}
    assert len(tr) <= 2 * 4 + 2 * 2 and len(tr) >= 2 * 4  # 1問あたり、4ターン（作図が4つを超えなければ）
    before = list(tr.index)
    tr.set_epoch(1)
    assert tr.index != before  # epoch が変わると、選ぶターンが変わる
    tr.set_epoch(0)
    assert tr.index == before  # 同じ epoch なら、同じ選び方
    ex = tr.example(*tr.index[0])
    assert ex["target"].endswith(data.END) and ex["id"].count("/") == 1
    assert (ex["image"] is not None) == ex["prompt"].startswith(data.IMAGE_MARK)


def test_info_delta_adds_only_new_numbers():
    prev = "<lengths> ab 1.0000 ; </lengths>"
    cur = "<lengths> ab 1.0000 ; ac 2.0000 ; </lengths> <angles> abc 90.0 ; </angles>"
    assert data.info_delta(prev, cur) == "<lengths> ac 2.0000 ; </lengths> <angles> abc 90.0 ; </angles>"
    assert data.info_delta(cur, cur) == "" and data.info_delta("", "") == ""


def test_sequence_has_every_turn_once_and_info_is_not_repeated():
    steps = expand({**to_compact(RECORDS[1]), "figure_points": [PTS] * 3})
    text, images = data.build_sequence(steps)
    assert len(images) == 3 and text.count(data.IMAGE_MARK) == 3  # 最初と、2つの作図のあと
    assert text.startswith(data.IMAGE_MARK + "\n<problem>") and "<history>" not in text
    assert text.count(data.THINK) == len(steps)
    assert all(s["output"] + data.END in text for s in steps)
    # 各ターンで足した数値を全部つなぐと、最後のターンの情報欄の数値と同じ
    got = [x for blk in text.split("<info> ")[1:] for x in data._INFO_ITEM.findall(blk.split(" </info>")[0])]
    assert sorted(got) == sorted(data._INFO_ITEM.findall(steps[-1]["info"]))
    assert len(got) == len(set(got))


def test_target_mask_covers_each_output_and_end_token():
    TE, IE = 5, 9  # </think>, <|im_end|>
    ids = [1, 2, 5, 3, 7, 7, 9, 4, 5, 3, 8, 9]
    m = data.target_mask(ids, TE, IE, 2)
    assert [i for i, x in enumerate(m) if x] == [4, 5, 6, 10, 11]


def test_target_mask_rejects_wrong_turn_count():
    import pytest

    with pytest.raises(ValueError):
        data.target_mask([5, 3, 7, 9], 5, 9, 2)
