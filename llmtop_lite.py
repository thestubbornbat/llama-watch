#!/usr/bin/env python3
"""
llmtop-lite -- a single-box inference monitor for llama.cpp server.

No dependencies. Python 3.8+. Renders with raw ANSI, reads the GPU via
nvidia-smi, and talks to llama.cpp's own HTTP endpoints.

Start your server with the monitoring endpoints enabled:

    llama-server -m model.gguf --metrics --slots --port 8080

Then:

    python3 llmtop_lite.py watch           # live dashboard
    python3 llmtop_lite.py dump            # raw /slots + /metrics (field discovery)
    python3 llmtop_lite.py probe           # fire one request, break down the latency

Field names in /slots and /metrics drift between llama.cpp releases. Everything
here degrades to "n/a" rather than crashing; run `dump` to see what your build
actually exposes and adapt SLOT_FIELDS below if something is missing.
"""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque

# ---------------------------------------------------------------- ANSI

CSI = "\x1b["
DIM, BOLD, RESET = CSI + "2m", CSI + "1m", CSI + "0m"
RED, YELLOW, GREEN, CYAN, GREY = (CSI + c + "m" for c in ("31", "33", "32", "36", "90"))


def hide_cursor():
    sys.stdout.write(CSI + "?25l")


def show_cursor():
    sys.stdout.write(CSI + "?25h")


def draw(lines):
    """Repaint in place without flicker: home, overwrite each line, clear rest."""
    out = [CSI + "H"]
    for ln in lines:
        out.append(ln + CSI + "K\n")
    out.append(CSI + "J")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def bar(frac, width=28, warn=0.75, crit=0.92):
    """Horizontal meter. frac is 0..1, or None for unknown."""
    if frac is None:
        return GREY + "?" * width + RESET
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    color = RED if frac >= crit else (YELLOW if frac >= warn else GREEN)
    return color + "#" * filled + RESET + GREY + "." * (width - filled) + RESET


def fmt(v, suffix="", nd=1):
    if v is None:
        return GREY + "n/a" + RESET
    if isinstance(v, float):
        return "{:.{}f}{}".format(v, nd, suffix)
    return "{}{}".format(v, suffix)


# ---------------------------------------------------------------- collectors


def http_get(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def parse_prometheus(text):
    """Flatten a Prometheus exposition into {metric_name: float}.

    Labels are dropped -- on a single box there is one server, so the base name
    is a sufficient key. This is deliberately name-agnostic: whatever your
    llama.cpp build exports shows up, so renamed metrics do not break anything.
    """
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name, raw = parts
        name = name.split("{", 1)[0].strip()
        try:
            out[name] = float(raw)
        except ValueError:
            continue
    return out


def get_gpu():
    """GPU counters via nvidia-smi. Returns None on a machine without one."""
    if not shutil.which("nvidia-smi"):
        return None
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    gpus = []
    for line in raw:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 5:
            continue

        def num(i):
            try:
                return float(f[i])
            except ValueError:
                return None

        gpus.append({
            "util": num(0), "mem_used": num(1), "mem_total": num(2),
            "temp": num(3), "power": num(4),
        })
    return gpus or None


# llama.cpp slot fields, in display order. Builds vary; missing keys render n/a.
SLOT_FIELDS = ("n_past", "n_ctx", "cache_tokens", "n_decoded", "n_prompt_tokens")


def slot_get(slot, *names):
    """First present key among names, searching one level into nested dicts."""
    for n in names:
        if n in slot and slot[n] is not None:
            return slot[n]
    for v in slot.values():
        if isinstance(v, dict):
            for n in names:
                if n in v and v[n] is not None:
                    return v[n]
    return None


# ---------------------------------------------------------------- rate tracking


class Rate:
    """Interval rate from monotonically increasing counters.

    Lifetime averages hide everything interesting. Dividing the token delta by
    the *server's own* seconds delta gives the rate while actually working,
    excluding idle time -- which is the number you care about.
    """

    def __init__(self):
        self.prev = {}

    def update(self, metrics, tok_key, sec_key):
        tok, sec = metrics.get(tok_key), metrics.get(sec_key)
        if tok is None or sec is None:
            return None
        key = tok_key
        prev = self.prev.get(key)
        self.prev[key] = (tok, sec)
        if prev is None:
            return None
        dt, ds = tok - prev[0], sec - prev[1]
        if ds <= 1e-9 or dt < 0:
            return None
        return dt / ds


# ---------------------------------------------------------------- watch


def find_metric(m, *candidates):
    for c in candidates:
        if c in m:
            return m[c]
    return None


def render(host, metrics, slots, gpus, rates, hist, show_prompts):
    L = []
    ts = time.strftime("%H:%M:%S")
    L.append(BOLD + "llmtop-lite" + RESET + GREY + "  " + host + "   " + ts + RESET)
    L.append("")

    # ---- GPU
    if gpus:
        for i, g in enumerate(gpus):
            mem_frac = None
            if g["mem_used"] is not None and g["mem_total"]:
                mem_frac = g["mem_used"] / g["mem_total"]
            L.append(BOLD + "GPU {}".format(i) + RESET)
            L.append("  util  {}  {}".format(
                bar((g["util"] or 0) / 100.0), fmt(g["util"], "%", 0)))
            L.append("  vram  {}  {} / {} MiB".format(
                bar(mem_frac), fmt(g["mem_used"], "", 0), fmt(g["mem_total"], "", 0)))
            L.append("  {}C   {}W".format(fmt(g["temp"], "", 0), fmt(g["power"], "", 0)))
    else:
        L.append(GREY + "GPU   no nvidia-smi on PATH" + RESET)
    L.append("")

    # ---- KV cache
    kv_ratio = find_metric(metrics, "llamacpp:kv_cache_usage_ratio", "kv_cache_usage_ratio")
    kv_tokens = find_metric(metrics, "llamacpp:kv_cache_tokens", "kv_cache_tokens")
    L.append(BOLD + "KV cache" + RESET)
    L.append("  used  {}  {}   tokens {}".format(
        bar(kv_ratio),
        fmt((kv_ratio * 100) if kv_ratio is not None else None, "%", 1),
        fmt(kv_tokens, "", 0)))
    L.append("")

    # ---- throughput
    prefill = rates.update(metrics, "llamacpp:prompt_tokens_total",
                           "llamacpp:prompt_seconds_total")
    decode = rates.update(metrics, "llamacpp:tokens_predicted_total",
                          "llamacpp:tokens_predicted_seconds_total")
    if decode is not None:
        hist.append(decode)
    processing = find_metric(metrics, "llamacpp:requests_processing", "requests_processing")
    deferred = find_metric(metrics, "llamacpp:requests_deferred", "requests_deferred")
    busy = find_metric(metrics, "llamacpp:n_busy_slots_per_decode")

    L.append(BOLD + "Throughput" + RESET + GREY + "  (interval, excludes idle)" + RESET)
    L.append("  prefill  {:>12}   decode  {:>12}".format(
        fmt(prefill, " tok/s"), fmt(decode, " tok/s")))
    L.append("  in-flight {:>3}   deferred {:>3}   busy slots/decode {}".format(
        fmt(processing, "", 0), fmt(deferred, "", 0), fmt(busy, "", 2)))
    if len(hist) > 1:
        lo, hi = min(hist), max(hist)
        L.append("  decode over last {} samples: {:.1f} - {:.1f} tok/s".format(
            len(hist), lo, hi))
    L.append("")

    # ---- slots
    L.append(BOLD + "Slots" + RESET)
    if slots is None:
        L.append(GREY + "  /slots unavailable -- start llama-server with --slots" + RESET)
    elif not slots:
        L.append(GREY + "  (none)" + RESET)
    else:
        L.append(GREY + "  id  state       n_past/n_ctx    cached   ctx used" + RESET)
        for s in slots:
            sid = slot_get(s, "id", "slot_id")
            processing_flag = slot_get(s, "is_processing")
            state = slot_get(s, "state")
            if processing_flag is not None:
                label = "PROCESSING" if processing_flag else "idle"
            elif state is not None:
                label = "PROCESSING" if state else "idle"
            else:
                label = "?"
            color = CYAN if label == "PROCESSING" else GREY

            n_past = slot_get(s, "n_past", "n_decoded")
            n_ctx = slot_get(s, "n_ctx")
            cached = slot_get(s, "cache_tokens", "n_cache_tokens", "cache_n")
            frac = (n_past / n_ctx) if (n_past is not None and n_ctx) else None

            L.append("  {:>2}  {}{:<10}{}  {:>6}/{:<6}  {:>6}   {}".format(
                sid if sid is not None else "?",
                color, label, RESET,
                fmt(n_past, "", 0), fmt(n_ctx, "", 0), fmt(cached, "", 0),
                bar(frac, width=16)))

            if show_prompts:
                p = slot_get(s, "prompt")
                if isinstance(p, str) and p:
                    L.append(GREY + "      " + p[:100].replace("\n", " ") + RESET)

    L.append("")
    if not show_prompts:
        L.append(GREY + "prompts redacted (--show-prompts to reveal)   q/ctrl-c to quit" + RESET)
    else:
        L.append(YELLOW + "prompts VISIBLE on screen" + RESET + GREY + "   ctrl-c to quit" + RESET)
    return L


def cmd_watch(args):
    host = args.host.rstrip("/")
    rates, hist = Rate(), deque(maxlen=60)
    hide_cursor()
    sys.stdout.write(CSI + "2J")
    try:
        while True:
            metrics = parse_prometheus(http_get(host + "/metrics"))
            raw_slots = http_get(host + "/slots")
            slots = None
            if raw_slots:
                try:
                    parsed = json.loads(raw_slots)
                    slots = parsed if isinstance(parsed, list) else parsed.get("slots")
                except json.JSONDecodeError:
                    slots = None
            if not metrics and slots is None:
                draw([RED + "cannot reach " + host + RESET, "",
                      "start it with:  llama-server -m model.gguf --metrics --slots",
                      GREY + "retrying..." + RESET])
            else:
                draw(render(host, metrics, slots, get_gpu(), rates, hist, args.show_prompts))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        sys.stdout.write("\n")


# ---------------------------------------------------------------- dump


def cmd_dump(args):
    host = args.host.rstrip("/")
    print("=== GET {}/slots ===".format(host))
    raw = http_get(host + "/slots")
    if raw is None:
        print("  unreachable (need --slots on the server)")
    else:
        try:
            print(json.dumps(json.loads(raw), indent=2)[:6000])
        except json.JSONDecodeError:
            print(raw[:6000])
    print()
    print("=== GET {}/metrics ===".format(host))
    m = parse_prometheus(http_get(host + "/metrics"))
    if not m:
        print("  unreachable (need --metrics on the server)")
    for k in sorted(m):
        print("  {:<48} {}".format(k, m[k]))


# ---------------------------------------------------------------- probe


def cmd_probe(args):
    """Fire one streaming request and attribute the latency.

    TTFT is measured client-side from the socket, which is the number a user
    actually feels. The server's own timings block is printed alongside it so
    you can see the gap between them (that gap is your transport overhead).
    """
    host = args.host.rstrip("/")
    body = json.dumps({
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "stream": True,
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        host + "/completion", data=body,
        headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    ttft = None
    token_times = []
    final = None

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw_line in r:
                line = raw_line.decode("utf-8", "replace").strip()
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
                    token_times.append(now)
                if chunk.get("stop") or "timings" in chunk:
                    final = chunk
    except urllib.error.URLError as e:
        print("request failed: {}".format(e))
        print("is llama-server running at {} ?".format(host))
        return 1

    total = time.perf_counter() - t0

    # inter-token latency from the stream itself
    itl = [ (token_times[i] - token_times[i-1]) * 1000.0
            for i in range(1, len(token_times)) ]
    itl.sort()

    def pct(p):
        if not itl:
            return None
        k = min(len(itl) - 1, int(round(p / 100.0 * (len(itl) - 1))))
        return itl[k]

    print(BOLD + "client-side (what the user feels)" + RESET)
    print("  TTFT              {}".format(fmt(ttft * 1000 if ttft else None, " ms", 1)))
    print("  total             {:.1f} ms".format(total * 1000))
    print("  tokens streamed   {}".format(len(token_times)))
    print("  inter-token p50   {}".format(fmt(pct(50), " ms", 1)))
    print("  inter-token p95   {}".format(fmt(pct(95), " ms", 1)))
    print("  inter-token p99   {}".format(fmt(pct(99), " ms", 1)))

    t = (final or {}).get("timings")
    if t:
        print()
        print(BOLD + "server-side" + RESET)
        pn, pms = t.get("prompt_n"), t.get("prompt_ms")
        dn, dms = t.get("predicted_n"), t.get("predicted_ms")
        print("  prefill           {} tok in {} ({})".format(
            fmt(pn, "", 0), fmt(pms, " ms", 1),
            fmt(t.get("prompt_per_second"), " tok/s", 1)))
        print("  decode            {} tok in {} ({})".format(
            fmt(dn, "", 0), fmt(dms, " ms", 1),
            fmt(t.get("predicted_per_second"), " tok/s", 1)))

        # Prompt cache hit rate is not exported anywhere -- derive it.
        cached = t.get("cache_n")
        if cached is None:
            cached = (final or {}).get("tokens_cached")
        if cached is not None and pn:
            hit = cached / float(pn + cached)
            print("  prompt cache      {} of {} tok reused  ({:.0%} hit)".format(
                fmt(cached, "", 0), fmt(pn + cached, "", 0), hit))
        else:
            print("  prompt cache      " + GREY + "not reported by this build" + RESET)

        if ttft is not None and pms is not None:
            overhead = ttft * 1000 - pms
            print("  transport gap     {:.1f} ms".format(overhead))
    else:
        print()
        print(GREY + "no timings in the final chunk -- use the native /completion"
              " endpoint, not /v1/chat/completions" + RESET)
    return 0


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="http://127.0.0.1:8080",
                   help="llama.cpp server base URL (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd")

    w = sub.add_parser("watch", help="live dashboard")
    w.add_argument("--interval", type=float, default=0.5)
    w.add_argument("--show-prompts", action="store_true",
                   help="reveal prompt text from /slots (off by default)")
    w.set_defaults(func=cmd_watch)

    d = sub.add_parser("dump", help="raw /slots and /metrics, for field discovery")
    d.set_defaults(func=cmd_dump)

    pr = sub.add_parser("probe", help="one request, full latency breakdown")
    pr.add_argument("--prompt", default="Explain KV caching in two sentences.")
    pr.add_argument("--n-predict", type=int, default=128)
    pr.add_argument("--timeout", type=float, default=120.0)
    pr.set_defaults(func=cmd_probe)

    args = p.parse_args()
    if not getattr(args, "func", None):
        args = p.parse_args(sys.argv[1:] + ["watch"])
    signal.signal(signal.SIGINT, lambda *a: (show_cursor(), sys.exit(0)))
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
