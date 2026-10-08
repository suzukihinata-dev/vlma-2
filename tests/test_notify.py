import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vlma import notify


class FakeSlack(BaseHTTPRequestHandler):
    calls: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeSlack.calls.append((self.path, self.headers.get("Authorization"), body))
        ok = {"ok": True, "ts": f"100.{len(FakeSlack.calls):03d}", "channel": body.get("channel")}
        if body.get("thread_ts") == "gone":
            ok = {"ok": False, "error": "thread_not_found"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(ok).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def slack(monkeypatch):
    FakeSlack.calls = []
    srv = HTTPServer(("127.0.0.1", 0), FakeSlack)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("SLACK_API_BASE", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL", "C123")
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    yield FakeSlack
    srv.shutdown()


def test_nothing_is_sent_without_configuration(monkeypatch, capsys):
    for k in ("SLACK_BOT_TOKEN", "SLACK_CHANNEL", "SLACK_WEBHOOK_URL"):
        monkeypatch.delenv(k, raising=False)
    assert notify.send("x") is False and "not configured" in capsys.readouterr().out


def test_first_message_becomes_the_parent_and_later_ones_reply_in_the_thread(slack, tmp_path):
    f = tmp_path / "run" / "slack_thread.json"
    assert notify.send("start", thread_file=f)
    assert notify.send("update 1", thread_file=f)
    assert notify.send("done", thread_file=f, broadcast=True)
    p0, p1, p2 = (c[2] for c in slack.calls)
    assert "thread_ts" not in p0  # 親
    assert p1["thread_ts"] == "100.001" and "reply_broadcast" not in p1  # 返信
    assert p2["thread_ts"] == "100.001" and p2["reply_broadcast"] is True  # 終了は、チャンネルにも表示
    assert slack.calls[0][1] == "Bearer xoxb-test" and p0["channel"] == "C123"
    assert json.loads(f.read_text())["ts"] == "100.001"


def test_the_thread_continues_after_a_restart_and_survives_a_deleted_parent(slack, tmp_path):
    f = tmp_path / "t.json"
    notify.send("a", thread_file=f)
    notify.send("b", thread_file=f)  # 再起動のあと（同じファイル）でも、同じスレッド
    assert slack.calls[1][2]["thread_ts"] == "100.001"
    f.write_text(json.dumps({"ts": "gone"}))  # 親が消された
    assert notify.send("c", thread_file=f)
    assert "thread_ts" not in slack.calls[-1][2]  # 新しい親になる
    assert json.loads(f.read_text())["ts"] == f"100.{len(slack.calls):03d}"


def test_messages_without_a_thread_file_are_plain_posts(slack):
    assert notify.send("x") and notify.send("y")
    assert all("thread_ts" not in c[2] for c in slack.calls)
