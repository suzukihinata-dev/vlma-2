"""Slack への通知。次のどちらかが環境変数にあるときだけ送る。

  1. SLACK_BOT_TOKEN と SLACK_CHANNEL   Bot User OAuth Token（xoxb-…）とチャンネル ID。chat:write の権限が要り、
                                  投稿先のチャンネルに、アプリを招待しておく。スレッドに積める（thread_file を指定したとき）。
  2. SLACK_WEBHOOK_URL            Incoming Webhook の URL。簡単だが、スレッドには積めない（常に新しい投稿になる）。
両方あれば、1 を使う。

スレッド: thread_file を渡すと、最初の投稿をスレッドの親にして、その ts を thread_file に保存する。以後の投稿は、その返信になる。
再起動しても、同じファイルを渡せば、同じスレッドに続く。親が消えていたら、新しい親を作る。
broadcast=True の返信は、チャンネルにも表示される（終了や警告に使う）。
トークンや URL はコードにも記録にも書かない。無ければ何もしない（標準出力に出すだけ）。標準ライブラリだけで動く。
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

_warned_webhook_thread = False


def _api() -> str:
    return os.environ.get("SLACK_API_BASE", "https://slack.com/api")


def bot_mode() -> bool:
    return bool(os.environ.get("SLACK_BOT_TOKEN")) and bool(os.environ.get("SLACK_CHANNEL"))


def configured() -> bool:
    return bot_mode() or bool(os.environ.get("SLACK_WEBHOOK_URL"))


def _post_api(payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        _api() + "/chat.postMessage", data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json; charset=utf-8", "Authorization": "Bearer " + os.environ["SLACK_BOT_TOKEN"]},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def send(text: str, *, thread_file: str | Path | None = None, broadcast: bool = False, timeout: float = 15.0) -> bool:
    """メッセージを送る。送れたら True。失敗しても例外は出さない（学習を止めないため）。"""
    global _warned_webhook_thread
    if not configured():
        print("[notify] Slack is not configured; message not sent:", text.replace("\n", " | ")[:200], flush=True)
        return False
    if not bot_mode():  # Webhook
        if thread_file and not _warned_webhook_thread:
            print("[notify] Incoming Webhook cannot post into threads; set SLACK_BOT_TOKEN and SLACK_CHANNEL to use threads.", flush=True)
            _warned_webhook_thread = True
        req = urllib.request.Request(os.environ["SLACK_WEBHOOK_URL"], data=json.dumps({"text": text}).encode(), method="POST",
                                     headers={"Content-Type": "application/json; charset=utf-8"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().strip() == b"ok"
        except Exception as e:  # noqa: BLE001
            print("[notify] Slack webhook failed:", type(e).__name__, flush=True)
            return False

    state = Path(thread_file) if thread_file else None
    parent = None
    if state and state.exists():
        try:
            parent = json.loads(state.read_text()).get("ts")
        except Exception:  # noqa: BLE001
            parent = None
    payload = {"channel": os.environ["SLACK_CHANNEL"], "text": text}
    if parent:
        payload["thread_ts"] = parent
        if broadcast:
            payload["reply_broadcast"] = True
    try:
        res = _post_api(payload, timeout)
        if not res.get("ok") and res.get("error") == "thread_not_found" and parent:  # 親が消えた: 新しい親を作る
            payload.pop("thread_ts"), payload.pop("reply_broadcast", None)
            parent, res = None, _post_api(payload, timeout)
    except Exception as e:  # noqa: BLE001
        print("[notify] Slack request failed:", type(e).__name__, flush=True)
        return False
    if not res.get("ok"):
        print("[notify] Slack API error:", res.get("error"), flush=True)  # 例: channel_not_found, not_in_channel, invalid_auth
        return False
    if state and not parent:  # この投稿が親になる
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"ts": res.get("ts"), "channel": res.get("channel")}))
    return True
