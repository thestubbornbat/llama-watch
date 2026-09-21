#!/usr/bin/env python3
"""Continuously generate streaming chat traffic for llmwatch.

Start llama-server on 8080 and llmwatch on its default proxy port (8081), then:

    python3 llmwatch_load.py

This intentionally targets port 8081 by default.  llmwatch only measures
requests which pass through its proxy; targeting llama-server's port 8080
directly bypasses the dashboard.
"""

import argparse
import http.client
import json
import signal
import sys
import threading
import time


STOP = threading.Event()


def request(host, port, path, prompt, max_tokens, timeout):
    """Send and fully consume one SSE request; return (elapsed_ms, chunks)."""
    body = json.dumps({
        "model": "local", "stream": True, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    started = time.perf_counter()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("POST", path, body=body, headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            "Accept": "text/event-stream",
        })
        response = conn.getresponse()
        if response.status < 200 or response.status >= 300:
            detail = response.read(500).decode("utf-8", "replace")
            raise RuntimeError("HTTP {}: {}".format(response.status, detail))
        chunks = 0
        for raw in response:
            if raw.startswith(b"data:") and raw.strip() != b"data: [DONE]":
                chunks += 1
        return (time.perf_counter() - started) * 1000.0, chunks
    finally:
        conn.close()


def worker(number, args, counter, lock):
    while not STOP.is_set():
        try:
            elapsed, chunks = request(args.host, args.port, args.path, args.prompt,
                                      args.max_tokens, args.timeout)
            with lock:
                counter["ok"] += 1
                counter["chunks"] += chunks
                counter["last_ms"] = elapsed
        except Exception as exc:
            with lock:
                counter["errors"] += 1
            print("worker {}: {}".format(number, exc), file=sys.stderr)
        STOP.wait(args.pause)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081,
                        help="llmwatch proxy port (default: 8081; use 8080 to bypass it)")
    parser.add_argument("--path", default="/v1/chat/completions",
                        help="generation endpoint (default: /v1/chat/completions)")
    parser.add_argument("--prompt", default="Write one short sentence about the ocean.")
    parser.add_argument("--prompt-repeat", type=int, default=64,
                        help="repeat the prompt to create a stable, larger prefix (default: 64)")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="simultaneous continuous clients")
    parser.add_argument("--pause", type=float, default=0.25,
                        help="seconds between each client's requests")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1 or args.prompt_repeat < 1 or args.pause < 0:
        parser.error("--concurrency, --max-tokens, and --prompt-repeat must be positive; "
                     "--pause cannot be negative")

    # Identical requests exercise llama.cpp's prompt cache.  A one-line prompt
    # is too small to make cache and prefill behaviour visible in the latency
    # meters, so use a moderate prefix by default.
    args.prompt = "\n\n".join([args.prompt] * args.prompt_repeat)

    counter, lock = {"ok": 0, "errors": 0, "chunks": 0, "last_ms": 0.0}, threading.Lock()
    threads = [threading.Thread(target=worker, args=(n + 1, args, counter, lock), daemon=True)
               for n in range(args.concurrency)]
    for thread in threads:
        thread.start()

    print("sending streaming requests to http://{}:{}{} with {} repeated prompt blocks "
          "(ctrl-c to stop)".format(args.host, args.port, args.path, args.prompt_repeat))
    try:
        while True:
            time.sleep(1)
            with lock:
                print("done={ok} errors={errors} SSE events={chunks} last={last_ms:.0f}ms".format(
                    **counter))
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        STOP.set()
        for thread in threads:
            thread.join(timeout=1)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    sys.exit(main())
