"""
Weight downloads survive dropped connections and never leave a truncated file in place.
Run from the server directory:
    python -m pytest tests -q
"""

import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import downloader.Downloader as D  # noqa: E402

PAYLOAD = os.urandom(300_000)


def _server(cut_after: int, honour_range: bool = True):
    """Serves PAYLOAD and drops the connection after cut_after bytes of each response."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            start = 0
            rng = self.headers.get("Range")
            if rng and honour_range:
                start = int(rng.split("=")[1].split("-")[0])
                self.send_response(206)
            else:
                self.send_response(200)
            body = PAYLOAD[start:]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:cut_after])
            self.wfile.flush()
            self.connection.shutdown(2)  # cut it, like the CDN did

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda s: None)


def test_resumes_after_dropped_connections(tmp_path):
    srv = _server(cut_after=50_000)
    dst = tmp_path / "w" / "model.onnx"
    D.download({"url": f"http://127.0.0.1:{srv.server_address[1]}/m", "saveTo": str(dst), "position": 0})
    assert dst.read_bytes() == PAYLOAD
    assert not (tmp_path / "w" / "model.onnx.part").exists()
    srv.shutdown()


def test_gives_up_without_leaving_a_truncated_model(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "MAX_STALLED_TRIES", 3)
    srv = _server(cut_after=10_000, honour_range=False)  # never gets past 10 kB
    dst = tmp_path / "model.onnx"
    D.download({"url": f"http://127.0.0.1:{srv.server_address[1]}/m", "saveTo": str(dst), "position": 0})
    assert not dst.exists()
    srv.shutdown()


def test_many_cuts_with_progress_keep_going(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "MAX_STALLED_TRIES", 3)
    srv = _server(cut_after=10_000)  # 30 cuts, each one advancing
    dst = tmp_path / "big.pt"
    D.download({"url": f"http://127.0.0.1:{srv.server_address[1]}/m", "saveTo": str(dst), "position": 0})
    assert dst.read_bytes() == PAYLOAD
    srv.shutdown()


def test_client_errors_are_not_retried(tmp_path):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(1)
            self.send_response(401)
            self.send_header("Content-Length", "29")
            self.end_headers()
            self.wfile.write(b"Invalid username or password.")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    dst = tmp_path / "rinna.pt"
    D.download({"url": f"http://127.0.0.1:{srv.server_address[1]}/m", "saveTo": str(dst), "position": 0})
    assert len(calls) == 1 and not dst.exists()
    srv.shutdown()


def test_tiny_model_files_count_as_missing(tmp_path):
    bad = tmp_path / "rinna_hubert_base_jp.pt"
    bad.write_bytes(b"Entry not found" + b" " * 14)  # 29 bytes, as found on a real install
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    assert not D.is_usable(str(bad))
    assert D.is_usable(str(cfg))
    assert not D.is_usable(str(tmp_path / "missing.onnx"))
