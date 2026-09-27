#!/usr/bin/env python3
"""End-to-end smoke test for llmwatch.

Starts a tiny llama.cpp-shaped upstream server, starts llmwatch in front of it,
then sends both streaming and non-streaming completion requests through the
proxy.  No llama.cpp installation, GPU, or third-party packages are required.

Run:
    python -m unittest discover -s tests -v
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class MockLlama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            data = b"\n".join((
                b"llamacpp:prompt_tokens_total 400",
                b"llamacpp:prompt_seconds_total 2",
                b"llamacpp:tokens_predicted_total 100",
                b"llamacpp:tokens_predicted_seconds_total 1",
                b"llamacpp:requests_deferred 0",
            )) + b"\n"
            self._reply(200, "text/plain", data)
        elif self.path == "/slots":
            data = json.dumps([{"id": 0, "is_processing": False,
                                "n_ctx": 8192, "n_prompt_tokens": 100,
                                "n_prompt_tokens_cache": 80,
                                "n_prompt_tokens_processed": 20}]).encode()
            self._reply(200, "application/json", data)
        else:
            self._reply(404, "text/plain", b"not found")

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # The production server terminates the response after [DONE].  Make
            # that boundary explicit here, otherwise HTTP/1.1 keeps the mock
            # connection alive and the proxy correctly continues reading it.
            self.send_header("Connection", "close")
            self.end_headers()
            events = [
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"timings": {"prompt_n": 20, "cache_n": 80,
                             "predicted_n": 2, "prompt_ms": 5}, "stop": True},
            ]
            for event in events:
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        else:
            data = json.dumps({"content": "plain response", "timings": {
                "prompt_n": 10, "cache_n": 0, "predicted_n": 2,
                "prompt_ms": 3}}).encode()
            self._reply(200, "application/json", data)

    def _reply(self, code, content_type, data):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def request(port, payload):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps(payload).encode()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json", "Content-Length": str(len(body))})
    response = conn.getresponse()
    data = response.read()
    assert response.status == 200, "proxy returned HTTP {}".format(response.status)
    assert response.getheader("Content-Length") is None, (
        "proxy must not forward Content-Length when it rechunks the response")
    assert response.getheader("Transfer-Encoding") == "chunked", (
        "proxy did not send a chunked downstream response")
    conn.close()
    return data


def wait_for_proxy(port, proc):
    deadline = time.time() + 5
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("llmwatch exited early")
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
            conn.request("GET", "/health")
            conn.getresponse().read()
            conn.close()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("llmwatch proxy did not start")


class ProxyEndToEndTest(unittest.TestCase):
    def test_proxy_preserves_responses_and_renders_metrics(self):
        self.assertTrue(
            os.path.isfile(os.path.join(SRC, "llamawatch", "__main__.py")),
            "cannot find the llamawatch package")

        upstream_port, proxy_port = free_port(), free_port()
        upstream = ThreadingHTTPServer(("127.0.0.1", upstream_port), MockLlama)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)

        proc = subprocess.Popen(
            [sys.executable, "-m", "llamawatch", "--listen", str(proxy_port),
             "--upstream-port", str(upstream_port), "--interval", "0.05",
             "--window", "60", "--ascii", "--no-color"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ,
                 "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")})

        def stop_proxy():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            return proc.communicate(timeout=5)

        wait_for_proxy(proxy_port, proc)
        streamed = request(proxy_port, {"stream": True, "prompt": "test"})
        plain = request(proxy_port, {"stream": False, "prompt": "test"})
        self.assertIn(b'"hello"', streamed, "SSE response changed")
        self.assertIn(b'" world"', streamed, "SSE response changed")
        self.assertEqual(json.loads(plain.decode())["content"], "plain response",
                          "plain response changed")
        time.sleep(0.15)  # allow the dashboard to render the completed records

        stdout, stderr = stop_proxy()
        self.assertIn(proc.returncode, (0, -signal.SIGINT),
                      "llmwatch failed:\n{}".format(stderr.decode("utf-8", "replace")))
        screen = stdout.decode("utf-8", "replace")
        for expected in ("PROMPT CACHE", "LATENCY", "CACHE IMPACT", "SLOTS & QUEUE",
                         "reuse", "SERVER", "RESPONSE SIZE", "80 reused"):
            self.assertIn(expected, screen, "dashboard never rendered {!r}: {}".format(
                expected, screen[-2000:]))


if __name__ == "__main__":
    unittest.main()
