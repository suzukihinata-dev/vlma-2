import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from vlma import actions as A
from vlma.data import build_index
from vlma.env import Episode
from vlma.envserver import Env, make_handler
from vlma.figures import figure_points, find_placement_seed, render_problem_figures
from vlma.steps import expand, to_compact

RECORDS = json.loads((Path(__file__).parent / "data" / "records.json").read_text())


def make_compact(tmp_path, rec):
    """学習データと同じ形（placement_seed と図の座標つき）のコンパクトな問題を、図ごと作る。"""
    c = to_compact(rec)
    seed = find_placement_seed(rec) or 0
    figs = render_problem_figures(rec, tmp_path / "data", seed)
    c["placement_seed"] = seed
    c["figure_points"] = [figure_points(p) for _, p in figs]
    c["n_turns"] = len(c["actions"])
    return c


@pytest.fixture(scope="module")
def compact(tmp_path_factory):
    return make_compact(tmp_path_factory.mktemp("env"), RECORDS[0])


def episode(compact, tmp_path, **kw):
    return Episode(compact, tmp_path / "data", tmp_path / "out", **kw)


def test_reference_replay_succeeds_with_shortest_proof(compact, tmp_path):
    ep = episode(compact, tmp_path)
    outs = [s["output"] for s in expand(compact)]
    for k, o in enumerate(outs):
        r = ep.step(o)
        assert r["verdict"]["ok"], (k, o, r["verdict"])
        assert r["done"] == (k == len(outs) - 1)
    res = ep.result()
    assert res["success"] and res["kind"] == "shortest" and res["rejected"] == 0
    assert res["accepted"] == len(outs) and res["needed"] == res["n_reference"]
    assert ep.history == outs  # 受理した手は、書式をそろえて残る（参照は、すでにその書式）


def test_state_matches_the_training_input_at_every_turn(compact, tmp_path):
    """環境が返す状態（情報欄・過去の手・図を見せるか）は、学習データの同じターンの入力と一致する。"""
    ep = episode(compact, tmp_path)
    samples = expand(compact)
    from vlma.data import shows_image

    for k, s in enumerate(samples):
        st = ep.state()
        assert st["info"] == s["info"] and st["history"] == s["history"], k
        assert (st["image"] is not None) == shows_image(samples, k), k
        ep.step(s["output"])


def test_construct_renders_a_new_figure(compact, tmp_path):
    ep = episode(compact, tmp_path)
    first = expand(compact)[0]["output"]
    assert first.startswith("CONSTRUCT")
    s0 = ep.state()
    assert Path(s0["image"]).exists()
    ep.step(first)
    s1 = ep.state()
    assert s1["image"] and s1["image"] != s0["image"] and Path(s1["image"]).exists()


def test_rejected_move_does_not_change_the_state(compact, tmp_path):
    ep = episode(compact, tmp_path)
    before = ep.state()
    r = ep.step("hello")
    assert not r["verdict"]["ok"] and r["verdict"]["reason"] == A.PARSE_FAILED and not r["done"]
    after = r["state"]
    assert {k: v for k, v in after.items() if k != "attempts"} == {k: v for k, v in before.items() if k != "attempts"}
    assert after["attempts"] == 1 and after["accepted"] == 0
    res = ep.result()
    assert not res["success"] and res["kind"] == "incomplete" and res["reject_reasons"] == {A.PARSE_FAILED: 1}


def test_episode_ends_when_the_attempt_budget_is_used_up(compact, tmp_path):
    ep = episode(compact, tmp_path, max_attempts=3)
    assert not ep.step("x")["done"] and not ep.step("x")["done"]
    assert ep.step("x")["done"]
    with pytest.raises(RuntimeError):
        ep.step("x")


def test_http_round_trip(compact, tmp_path):
    d = tmp_path / "srvdata"
    d.mkdir()
    (d / "problems.jsonl").write_text(json.dumps(compact) + "\n")
    build_index(d)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Env(str(d), str(tmp_path / "srvout"))))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    try:
        r = post("/reset", {"problem_id": compact["problem_id"]})
        eid = r["episode"]
        assert r["state"]["problem"] == compact["problem"] and r["n_reference"] > 0
        for o in [s["output"] for s in expand(compact)]:
            out = post("/step", {"episode": eid, "output": o})
            assert out["verdict"]["ok"]
        assert out["done"] and out["verdict"]["goal_reached"]
        assert post("/close", {"episode": eid})["result"]["success"]
        assert json.loads(urllib.request.urlopen(base + "/health").read())["ok"]
    finally:
        srv.shutdown()


def test_prompts_match_the_training_data_for_both_formats(compact, tmp_path):
    """環境の状態から作るプロンプトは、学習データ（方式 A' の各ターン、方式 B の系列）と同じ文章になる。"""
    from vlma import bench, data

    ep = episode(compact, tmp_path)
    samples = expand(compact)
    turn, seq = bench.TurnPrompt(), bench.SeqPrompt()
    for k, s in enumerate(samples):
        st = ep.state()
        p, imgs = turn.prompt(st)
        want, _ = data.build_text(s, with_image=data.shows_image(samples, k))
        assert p == want and len(imgs) == int(data.shows_image(samples, k)), k
        sp, _ = seq.prompt(st)
        assert sp == seq.prompt(st)[0]  # prompt は状態を変えない
        ep.step(s["output"])
        seq.accept(st, s["output"])
        turn.accept(st, s["output"])
    full, images = data.build_sequence(samples)
    assert seq.text == full and len(seq.images) == len(images)


def test_summarize_counts_everything():
    from vlma import bench

    r = lambda **kw: {"problem_id": "p", "success": False, "kind": "incomplete", "n_reference": 5, "n_reference_turns": 8, "attempts": 4, "accepted": 3,
                      "rejected": 1, "reject_reasons": {"not_hold": 1}, "first_reject_attempt": 3, "needed": 2, "unneeded": 0, "reward_total": -1.0, **kw}
    s = bench.summarize([r(), r(success=True, kind="shortest", rejected=0, reject_reasons={}, first_reject_attempt=None, accepted=8, attempts=8,
                              needed=5, reward_total=5.0)])
    assert s["problems"] == 2 and s["success_rate"] == 0.5 and s["kind_rate"]["shortest"] == 0.5
    assert s["reject_reasons"] == {"not_hold": 1} and s["move_accept_rate"] == round(11 / 12, 4)
    assert s["progress"] == round((3 / 8 + 1) / 2, 4) and s["success_by_reference_length"]["turns<=20"]["n"] == 2


def test_solver_crash_is_a_rejection_not_a_server_error(compact, tmp_path, monkeypatch):
    """DDAR の実装が退化した図で例外を出しても、環境は落ちず、却下（not_hold）として返す。"""
    ep = episode(compact, tmp_path)
    ep.step(expand(compact)[0]["output"])  # 作図を受理

    def boom(self, engine):
        raise AttributeError("'Line' object has no attribute 'points'")

    monkeypatch.setattr(type(ep.verifier), "_saturated_proof", boom)
    r = ep.step("APPLY Angle_chasing : eqangle a b a c b d c d [900] from [000]")
    assert not r["verdict"]["ok"] and r["verdict"]["reason"] == A.NOT_HOLD and "AttributeError" in r["verdict"]["detail"]
    assert not r["done"] and r["state"]["accepted"] == 1


def test_summarize_reports_coverage_and_success_within_budget():
    from vlma import bench

    base = {"problem_id": "p", "success": False, "kind": "incomplete", "n_reference": 4, "n_reference_turns": 10, "attempts": 10, "accepted": 6,
            "rejected": 4, "reject_reasons": {"not_hold": 4}, "first_reject_attempt": 1, "needed": 2, "unneeded": 0, "reward_total": 0.0}
    solved = dict(base, success=True, kind="wasteful", attempts=14, accepted=10, rejected=4, needed=4)
    s = bench.summarize([base, solved])
    assert s["reference_coverage"] == round((0.5 + 1.0) / 2, 4)
    assert s["success_within"] == {"1x_reference_turns": 0.0, "1.5x_reference_turns": 0.5, "2x_reference_turns": 0.5}
