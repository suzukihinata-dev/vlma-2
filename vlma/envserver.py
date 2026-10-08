"""ベンチマークの環境サーバー（HTTP、標準ライブラリだけ）。genesisgeo のコンテナで動かし、学習のコンテナのクライアントから使う。

usage: python -m vlma.envserver --port 8100 --data-dir /artifacts/genesisgeo/r2/build-100k --out-dir /artifacts/bench/<name>
  POST /reset {"problem_id": ..., "mode": "full"|"construct", "max_attempts": n, "max_constructs": n}
                                                 -> {"episode": id, "state": {...}, "n_reference": n}（state.done が真なら、作図なしで解けた）
  POST /step  {"episode": id, "output": text}    -> {"verdict": {...}, "done": bool, "state": {...}}
  POST /close {"episode": id}                    -> {"result": {...}}（エピソードを閉じて、結果を返す）
  GET  /health                                   -> {"ok": true}
DDAR の計算は CPU で、プロセス内のスレッドでは並列にならない。並列にするときは、ポートを変えて複数のプロセスを起動する
（scripts/bench_env.sh）。1つのエピソードは、同じプロセスの中だけで続ける。
"""
from __future__ import annotations

import argparse
import itertools
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from vlma.env import Episode


class Env:
    def __init__(self, data_dir: str, out_dir: str):
        self.data_dir, self.out_dir = Path(data_dir), Path(out_dir)
        self.index = json.load(open(self.data_dir / "turn_index.json"))
        self.episodes: dict[str, Episode] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._file = threading.local()

    def _compact(self, pid: str) -> dict:
        f = getattr(self._file, "f", None)
        if f is None:
            f = self._file.f = open(self.data_dir / "problems.jsonl", "rb")
        f.seek(self.index[pid][0])
        return json.loads(f.readline())

    def reset(self, pid: str, max_attempts: int | None = None, mode: str = "full", max_constructs: int = 4) -> dict:
        ep = Episode(self._compact(pid), self.data_dir, self.out_dir, max_attempts, mode, max_constructs)
        with self._lock:
            eid = f"{pid}-{next(self._ids)}"
            self.episodes[eid] = ep
        return {"episode": eid, "state": ep.state(), "n_reference": ep.n_reference, "n_reference_turns": len(ep.reference)}

    def step(self, eid: str, output: str) -> dict:
        return self.episodes[eid].step(output)

    def close(self, eid: str) -> dict:
        with self._lock:
            ep = self.episodes.pop(eid)
        return {"result": ep.result()}


def make_handler(env: Env):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "episodes": len(env.episodes)}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/reset":
                    out = env.reset(body["problem_id"], body.get("max_attempts"), body.get("mode", "full"), body.get("max_constructs", 4))
                elif self.path == "/step":
                    out = env.step(body["episode"], body["output"])
                elif self.path == "/close":
                    out = env.close(body["episode"])
                else:
                    return self._send(404, {"error": "not found"})
                self._send(200, out)
            except Exception as e:  # noqa: BLE001  クライアントに理由を返す（学習側を落とさない）
                self._send(500, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--data-dir", default="/artifacts/genesisgeo/r2/build-100k")
    ap.add_argument("--out-dir", default="/artifacts/bench/figs-live")
    a = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(Env(a.data_dir, a.out_dir))).serve_forever()


if __name__ == "__main__":
    main()
