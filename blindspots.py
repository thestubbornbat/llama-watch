#!/usr/bin/env python3
"""
blindspots.py -- prove what a GPU monitor cannot see.

Runs four controlled experiments against a llama.cpp server while sampling
nvidia-smi in the background. Each experiment changes latency a lot. The GPU
counters barely move. That contrast is the whole point.

    llama-server -m model.gguf --metrics --slots -np 4 -c 8192 --port 8080
    python3 blindspots.py

No dependencies. Python 3.8+.

Experiments
  1 cold vs warm      same prompt twice -- the prompt cache appears
  2 prefix vs suffix  edit one word at the START vs the END of an identical
                      prompt. Same token count, same everything a GPU monitor
                      can see. Wildly different work.
  3 read vs write     long-prompt/short-answer vs short-prompt/long-answer,
                      matched total tokens. Time lands somewhere different.
  4 concurrency tail  N parallel requests. Watch p99 detach from p50 while
                      utilization looks fine.
"""

import argparse
import json
import statistics
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BOLD, DIM, RESET = "\x1b[1m", "\x1b[2m", "\x1b[0m"
CYAN, YELLOW, GREEN, GREY = "\x1b[36m", "\x1b[33m", "\x1b[32m", "\x1b[90m"

HOST = "http://127.0.0.1:8080"


# ------------------------------------------------------------------ plumbing


def post(path, payload, timeout=180.0, stream=False):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(text):
    """Exact token count via the server, so experiments are properly matched."""
    try:
        return len(post("/tokenize", {"content": text}).get("tokens", []))
    except Exception:
        return max(1, len(text.split()))  # crude fallback


def slot_ctx(default=4096):
    """Per-slot context budget.

    This matters and is easy to get wrong: llama.cpp's -c is the TOTAL KV
    budget split across -np slots, so `-c 8192 -np 4` gives each slot 2048.
    A prompt larger than that is rejected with HTTP 400, not truncated.
    """
    try:
        raw = urllib.request.urlopen(HOST + "/slots", timeout=3.0).read()
        s = json.loads(raw.decode("utf-8", "replace"))
        s = s if isinstance(s, list) else s.get("slots", [])
        v = s[0].get("n_ctx")
        return int(v) if v else default
    except Exception:
        return default


def erase_slot(slot_id=0):
    """Clear a slot's cached prompt so 'cold' really is cold."""
    try:
        post("/slots/{}?action=erase".format(slot_id), {}, timeout=10.0)
        return True
    except Exception:
        return False


class GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background. This is the control group.

    Caveat that matters: nvidia-smi's utilization.gpu is itself a windowed
    average over the driver's own internal sample period (often ~1s), so it
    LAGS. A request shorter than that window produces a reading contaminated by
    whatever ran before it. We report the sample count so a reading taken over
    too few samples can be discarded rather than believed.
    """

    def __init__(self, period=0.15):
        super().__init__(daemon=True)
        self.samples = []
        self.period = period
        self._stop = threading.Event()
        self.available = bool(shutil.which("nvidia-smi"))

    def run(self):
        if not self.available:
            return
        q = "utilization.gpu,memory.used,power.draw"
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=" + q,
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0).stdout
                f = [x.strip() for x in out.strip().splitlines()[0].split(",")]
                self.samples.append((float(f[0]), float(f[1]), float(f[2])))
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if not self.samples:
            return None
        return {
            "util": statistics.mean(s[0] for s in self.samples),
            "util_max": max(s[0] for s in self.samples),
            "vram": statistics.mean(s[1] for s in self.samples),
            "power": statistics.mean(s[2] for s in self.samples),
            "n": len(self.samples),
            # fewer than 3 samples cannot describe a window; say so instead of lying
            "reliable": len(self.samples) >= 3,
        }


def run_request(prompt, n_predict, cache=True, timeout=180.0):
    """One streaming request. Returns client-measured latency + server timings."""
    t0 = time.perf_counter()
    ttft, marks, final = None, [], None
    body = {"prompt": prompt, "n_predict": n_predict,
            "stream": True, "cache_prompt": cache, "temperature": 0.0}
    try:
        resp = post("/completion", body, timeout=timeout, stream=True)
    except urllib.error.URLError as e:
        return {"error": str(e)}
    with resp as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            now = time.perf_counter()
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if chunk.get("content"):
                if ttft is None:
                    ttft = now - t0
                marks.append(now)
            if chunk.get("stop") or "timings" in chunk:
                final = chunk

    itl = sorted((marks[i] - marks[i - 1]) * 1000.0 for i in range(1, len(marks)))
    t = (final or {}).get("timings") or {}
    cached = t.get("cache_n")
    if cached is None:
        cached = (final or {}).get("tokens_cached")
    return {
        "ttft_ms": (ttft * 1000.0) if ttft else None,
        "total_ms": (time.perf_counter() - t0) * 1000.0,
        "n_tokens": len(marks),
        "itl": itl,
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "cached": cached,
    }


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def f(v, nd=1):
    return ("{:.%df}" % nd).format(v) if isinstance(v, (int, float)) else "n/a"


# ------------------------------------------------------------------ prompts


FILLER = (
    "The support agent reviewed the customer account history carefully before "
    "responding to the billing question raised during the previous call. ")


def build_prompt(target_tokens, head="Context A.", tail="Summarize briefly."):
    """A prompt of roughly target_tokens, with a distinct head and tail."""
    body = ""
    while tokenize(head + " " + body + " " + tail) < target_tokens:
        body += FILLER
        if len(body) > 200000:
            break
    return "{} {} {}".format(head, body, tail)


# ------------------------------------------------------------------ experiments


def banner(n, title, question):
    print()
    print(BOLD + "[{}] {}".format(n, title) + RESET)
    print(GREY + "    " + question + RESET)


def with_gpu(fn, cooldown=1.2):
    """Run fn while sampling the GPU, sampling ONLY during fn.

    The earlier version padded the window with idle time on both sides, which
    dragged the mean toward zero by a different amount for fast and slow
    requests -- manufacturing a difference that was pure measurement artifact.
    The cooldown lets the driver's lagging counter drain from the previous run.
    """
    time.sleep(cooldown)
    g = GpuSampler()
    g.start()
    out = fn()
    return out, g.stop()


_NONCE = [0]


def nonce():
    """Unique string, so a 'cold' prompt is genuinely uncached without relying
    on /slots erase (which fails silently on some builds)."""
    _NONCE[0] += 1
    return "Session {}-{}.".format(int(time.time() * 1000) % 100000, _NONCE[0])


def exp1_cold_warm(args):
    banner(1, "Cold vs warm", "identical prompt, twice. Does the GPU look different?")
    # A unique head guarantees this prompt has never been seen, so "cold" is
    # cold on every run -- not only the first one after server start.
    p = build_prompt(args.prompt_tokens, head=nonce())
    erase_slot()
    rows = []
    for label in ("cold", "warm"):
        r, gpu = with_gpu(lambda: run_request(p, 32))
        if r.get("error"):
            print("    request failed:", r["error"])
            return None
        rows.append((label, r, gpu))
    return rows


def exp2_prefix_suffix(args):
    banner(2, "Prefix vs suffix edit",
           "same length, same tokens. One word moved -- start or end?")
    tag = nonce()  # isolates this run from every previous run's cache
    base = build_prompt(args.prompt_tokens, head=tag + " Context A.", tail="Summarize briefly.")
    suffix_edit = build_prompt(args.prompt_tokens, head=tag + " Context A.", tail="Summarize concisely.")
    prefix_edit = build_prompt(args.prompt_tokens, head=tag + " Context B.", tail="Summarize briefly.")

    erase_slot()
    run_request(base, 16)  # warm the cache on `base`
    rows = []
    for label, p in (("edit at END", suffix_edit), ("edit at START", prefix_edit)):
        r, gpu = with_gpu(lambda p=p: run_request(p, 32))
        if r.get("error"):
            print("    request failed:", r["error"])
            return None
        rows.append((label, r, gpu))
        # re-warm on base so both variants face the same cache state
        run_request(base, 16)
    return rows


def report_split(rows):
    """For experiment 3 the interesting axis is not TTFT, it is where time went."""
    if not rows:
        return
    print()
    print(GREY + "    {:<24} {:>11} {:>11} {:>9} {:>14}".format(
        "", "prefill ms", "decode ms", "total ms", "time in decode") + RESET)
    shares = []
    for label, r, _gpu in rows:
        pre = r.get("prompt_ms") or 0.0
        dec = r.get("predicted_ms") or 0.0
        tot = pre + dec
        share = (dec / tot * 100.0) if tot else None
        if share is not None:
            shares.append(share)
        print("    {:<24} {:>11} {:>11} {:>9} {:>13}%".format(
            label, f(pre), f(dec), f(tot), f(share, 0)))
    print()
    # State the conclusion only if the data actually supports it. A harness
    # that prints its thesis regardless of the numbers is worse than useless.
    if len(shares) >= 2 and abs(shares[0] - shares[-1]) >= 15:
        print("    {}Same machine, same model -- but the time lands in completely "
              "different\n    places ({}% vs {}% in decode). A single tok/s number "
              "averages this away.{}".format(
                  YELLOW, f(shares[0], 0), f(shares[-1], 0), RESET))
    else:
        print("    {}No meaningful split difference here. Either the prompt was "
              "cached (check\n    the prefill column -- near zero means cached, "
              "not fast) or the two cases\n    were not actually different "
              "enough. Try a larger --prompt-tokens.{}".format(GREY, RESET))


def exp3_read_write(args):
    banner(3, "Reading vs writing",
           "matched total tokens, opposite split. Where does the time go?")
    # Both prompts get a unique head AND run with cache_prompt=False. Relying
    # on /slots erase was the bug: it fails silently on some builds, so the
    # "long prompt" case was served entirely from cache and measured nothing.
    ctx = slot_ctx()
    want = args.prompt_tokens * 2
    long_n = min(want, int(ctx * 0.55))
    if long_n < want:
        print(GREY + "    (clamped long prompt {} -> {} tokens; each slot only "
              "has {} of context)".format(want, long_n, ctx) + RESET)
    long_p = build_prompt(long_n, head=nonce())
    short_p = nonce() + " Write a paragraph about batteries."
    out_n = max(96, min(args.prompt_tokens // 3, int(ctx * 0.3)))
    rows = []
    for label, p, n in (("long prompt / 32 out", long_p, 32),
                        ("short prompt / {} out".format(out_n), short_p, out_n)):
        r, gpu = with_gpu(lambda p=p, n=n: run_request(p, n, cache=False))
        if r.get("error"):
            print("    request failed:", r["error"])
            return None
        rows.append((label, r, gpu))
    return rows


def exp4_concurrency(args):
    banner(4, "Concurrency tail",
           "{} requests, mixed sizes, oversubscribed. Watch p99 leave p50."
           .format(args.requests))

    # The old version fired N identical requests into N slots. Identical work
    # arriving simultaneously and finishing simultaneously has no tail BY
    # CONSTRUCTION -- p50 and p99 were the same number because they described
    # the same request. A tail needs queueing (more requests than slots) and
    # heterogeneous cost (mixed prompt lengths).
    ctx = slot_ctx()
    base = args.prompt_tokens
    cap = int(ctx * 0.45)  # leave room for the generated tokens
    sizes = [min(cap, s) for s in
             (max(64, base // 4), max(96, base // 2), base, base * 2)]
    if max(sizes) < base * 2:
        print(GREY + "    (prompt sizes capped at {} tokens; each slot has {} "
              "of context)".format(cap, ctx) + RESET)
    bodies = [build_prompt(s, head="Body{}.".format(i)) for i, s in enumerate(sizes)]

    # Uncontended baseline first, so "under load" has something to be worse than.
    solo, _ = with_gpu(lambda: run_request(nonce() + " " + bodies[2], 32, cache=False))
    solo_ttft = solo.get("ttft_ms")

    def fire():
        out = [None] * args.requests
        threads = []
        for i in range(args.requests):
            body = bodies[i % len(bodies)]

            def work(i=i, body=body):
                out[i] = run_request(nonce() + " " + body, 48, cache=False)

            threads.append(threading.Thread(target=work))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return out

    raw, gpu = with_gpu(fire)
    results = [o for o in raw if o and not o.get("error")]
    errors = [o["error"] for o in raw if o and o.get("error")]
    if errors:
        # Dropping failures silently biases the tail: the requests that fail
        # are the expensive ones, so the survivors look better than reality.
        seen = {}
        for e in errors:
            seen[e] = seen.get(e, 0) + 1
        print()
        for e, n in seen.items():
            print("    {}{} request(s) FAILED: {}{}".format(YELLOW, n, e, RESET))
        print("    {}those were the largest prompts -- the numbers below are "
              "survivors only,\n    so the real tail is worse than shown.{}"
              .format(GREY, RESET))
    if not results:
        print("    all requests failed")
        return None

    ttfts = sorted(r["ttft_ms"] for r in results if r["ttft_ms"])
    all_itl = sorted(v for r in results for v in r["itl"])
    print()
    print("    completed          {} / {}".format(len(results), args.requests))
    print("    TTFT solo          {} ms   {}(uncontended baseline){}".format(
        f(solo_ttft), GREY, RESET))
    for p in (50, 95, 99):
        print("    TTFT p{:<2}           {} ms".format(p, f(pct(ttfts, p))))
    print("    TTFT max           {} ms".format(f(ttfts[-1] if ttfts else None)))
    print("    inter-token p50    {} ms".format(f(pct(all_itl, 50))))
    print("    inter-token p99    {} ms".format(f(pct(all_itl, 99))))

    p50, p99 = pct(ttfts, 50), pct(ttfts, 99)
    if p50 and p99:
        print()
        print("    {}tail ratio p99/p50   {}x{}".format(YELLOW, f(p99 / p50, 2), RESET))
    if solo_ttft and p99:
        print("    {}p99 vs solo          {}x worse under load{}".format(
            YELLOW, f(p99 / solo_ttft, 2), RESET))
    if gpu:
        tag = "" if gpu["reliable"] else "  (too few samples -- unreliable)"
        print("    {}GPU util mean {}%  max {}%{}{}".format(
            GREY, f(gpu["util"], 0), f(gpu["util_max"], 0), tag, RESET))
    return None


# ------------------------------------------------------------------ reporting


def report(rows):
    """Side-by-side table. The rightmost column is the punchline."""
    if not rows:
        return
    print()
    hdr = "    {:<22} {:>10} {:>11} {:>10} {:>9} {:>11}".format(
        "", "TTFT ms", "prefill ms", "reused", "dec tok/s", "GPU util %")
    print(GREY + hdr + RESET)
    base_ttft = None
    for label, r, gpu in rows:
        dec = None
        if r.get("predicted_n") and r.get("predicted_ms"):
            dec = r["predicted_n"] / (r["predicted_ms"] / 1000.0)
        if gpu and gpu["reliable"]:
            util = f(gpu["util"], 0)
        elif gpu:
            util = "~" + f(gpu["util"], 0)  # too few samples to trust
        else:
            util = "n/a"
        line = "    {:<22} {:>10} {:>11} {:>10} {:>9} {:>11}".format(
            label, f(r.get("ttft_ms")), f(r.get("prompt_ms")),
            f(r.get("cached"), 0), f(dec), util)
        print(line)
        if base_ttft is None:
            base_ttft = r.get("ttft_ms")

    a, b = rows[0][1].get("ttft_ms"), rows[-1][1].get("ttft_ms")
    if a and b:
        ratio = max(a, b) / min(a, b)
        print()
        print("    {}TTFT differs by {}x.{}".format(YELLOW, f(ratio, 2), RESET), end="")
        ga, gb = rows[0][2], rows[-1][2]
        if ga and gb:
            note = "" if (ga["reliable"] and gb["reliable"]) else \
                "  <- a '~' reading is fewer than 3 samples; the request was " \
                "shorter than\n       nvidia-smi's own averaging window, so " \
                "that number is noise, not signal."
            print(" {}GPU util: {}% vs {}%.{}{}".format(
                GREY, f(ga["util"], 0), f(gb["util"], 0), note, RESET))
        else:
            print(GREY + " (no GPU samples -- nvidia-smi not found)" + RESET)


def main():
    global HOST
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--prompt-tokens", type=int, default=1024,
                    help="size of the long test prompt (default: %(default)s)")
    ap.add_argument("--parallel", type=int, default=4,
                    help="server slot count, for reference (set -np to match)")
    ap.add_argument("--requests", type=int, default=24,
                    help="requests fired at once in experiment 4; must exceed "
                         "the server's -np or there is no queueing and no tail")
    ap.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                    help="run a single experiment")
    args = ap.parse_args()
    HOST = args.host.rstrip("/")

    try:
        urllib.request.urlopen(HOST + "/health", timeout=3.0)
    except Exception:
        try:
            urllib.request.urlopen(HOST + "/slots", timeout=3.0)
        except Exception:
            print("cannot reach {}.".format(HOST))
            print("start it with:  llama-server -m model.gguf --metrics --slots -np 4")
            return 1

    if not shutil.which("nvidia-smi"):
        print(GREY + "note: nvidia-smi not found -- latency still measured, "
              "GPU control column will read n/a" + RESET)

    print(BOLD + "blindspots" + RESET + GREY + "  " + HOST + RESET)
    print(GREY + "  each experiment changes latency a lot. watch the GPU column." + RESET)

    for n, fn in ((1, exp1_cold_warm), (2, exp2_prefix_suffix),
                  (3, exp3_read_write), (4, exp4_concurrency)):
        if args.only and args.only != n:
            continue
        try:
            (report_split if n == 3 else report)(fn(args))
        except KeyboardInterrupt:
            print("\ninterrupted")
            return 1
        except Exception as e:
            print("    experiment {} failed: {}".format(n, e))

    print()
    print(GREY + "if the GPU column is flat while TTFT moves, that is the gap." + RESET)
    return 0


if __name__ == "__main__":
    sys.exit(main())
