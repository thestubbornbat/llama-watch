#!/usr/bin/env python3
"""
llmwatch -- live monitor for llama.cpp that sees what /metrics cannot.

Existing llama.cpp dashboards render /metrics + /slots. Those endpoints expose
server-level aggregates and slot state, which means the two numbers that
actually predict user experience are simply absent:

  * prompt-cache hit rate   -- decides whether a turn costs 9ms or 1.4s
  * inter-token p99         -- the mid-response stall a user hears as a freeze

Both are per-request properties. So llmwatch sits in front of the server as a
transparent streaming proxy, measures every request as it passes, and renders
those distributions next to the KV cache and GPU state.

    llama-server -m model.gguf --metrics --slots -np 4 -c 8192 --port 8080
    python3 llmwatch.py                      # proxy on 8081, TUI in this terminal

Then point your application at port 8081 instead of 8080. Nothing else changes.

No dependencies. Python 3.8+.
"""

import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .collectors import RATES, get_gpu, http_get, mget, parse_prom
from .config import build_parser

CSI = "\x1b["
BOLD, DIM, RESET = CSI + "1m", CSI + "2m", CSI + "0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Theme:
    """Color with graceful degradation, and never color alone.

    Status is always accompanied by a text label, because a terminal may be
    monochrome, piped, or read by someone who cannot distinguish the hues.
    Chrome (borders, units) is recessive; data is the only thing that is bright.
    """

    def __init__(self, ascii_only=False, force_mono=False):
        self.ascii = ascii_only
        term = os.environ.get("TERM", "")
        ct = os.environ.get("COLORTERM", "")
        self.mono = (force_mono or os.environ.get("NO_COLOR") is not None
                     or term in ("", "dumb"))
        self.truecolor = (not self.mono) and ct in ("truecolor", "24bit")

    def _c(self, rgb, x256):
        if self.mono:
            return ""
        if self.truecolor:
            return CSI + "38;2;{};{};{}m".format(*rgb)
        return CSI + "38;5;{}m".format(x256)

    # status palette -- reserved, never reused for anything else
    @property
    def good(self):
        return self._c((111, 168, 111), 71)

    @property
    def warn(self):
        return self._c((201, 150, 47), 179)

    @property
    def crit(self):
        return self._c((194, 91, 91), 167)

    @property
    def accent(self):
        return self._c((90, 140, 168), 67)

    # text tokens -- values and labels wear these, never a status hue
    @property
    def ink(self):
        return "" if self.mono else RESET

    @property
    def muted(self):
        return self._c((150, 150, 150), 245)

    @property
    def chrome(self):
        return self._c((110, 110, 110), 240)

    @property
    def off(self):
        return "" if self.mono else RESET

    def bold(self, s):
        return s if self.mono else BOLD + s + RESET


T = Theme()


def vlen(s):
    """Visible width, ignoring escape sequences."""
    return len(ANSI_RE.sub("", s))


def clip(s, w):
    """Keep ANSI styling while clipping a line to a terminal cell width.

    Data values can grow arbitrarily (large token counts, model names, or a
    narrow SSH terminal).  Panels must be a hard visual boundary, not a best
    effort.  The renderer only uses single-cell glyphs, so character width is
    sufficient here.
    """
    if w <= 0:
        return ""
    if vlen(s) <= w:
        return s
    out, pos, used = [], 0, 0
    while pos < len(s) and used < w:
        match = ANSI_RE.match(s, pos)
        if match:
            out.append(match.group(0))
            pos = match.end()
        else:
            out.append(s[pos])
            pos += 1
            used += 1
    # A reset prevents an intentionally clipped coloured value from tinting a
    # border or the next panel row.
    return "".join(out) + (RESET if ANSI_RE.search(s) else "")


def pad(s, w):
    s = clip(s, w)
    return s + " " * max(0, w - vlen(s))


def num(v, nd=0):
    """Thousands-separated. Alignment is most of readability."""
    if v is None:
        return "--"
    return "{:,.{}f}".format(v, nd)


# ---------------------------------------------------------------- marks

BLOCKS = " ▏▎▍▌▋▊▉"   # 1/8 .. 7/8
FULL = "█"
TRACK = "░"
SPARK = "▁▂▃▄▅▆▇█"


def meter(frac, width, tone):
    """Magnitude bar with 1/8-cell resolution. One hue, light->dark by fill."""
    if frac is None:
        return T.chrome + ("?" * width if T.ascii else TRACK * width) + T.off
    frac = max(0.0, min(1.0, frac))
    if T.ascii:
        n = int(round(frac * width))
        if frac > 0 and n == 0:
            n = 1
        return tone + "#" * n + T.chrome + "-" * (width - n) + T.off
    exact = frac * width
    full = int(exact)
    rem = int((exact - full) * 8)
    # A real, non-zero value should not be visually indistinguishable from
    # zero just because it is below one eighth of a terminal cell.
    if frac > 0 and full == 0 and rem == 0:
        rem = 1
    bar = FULL * full + (BLOCKS[rem] if rem and full < width else "")
    return tone + bar + T.chrome + TRACK * (width - vlen(bar)) + T.off


def spark(vals, width=12):
    """Trend. Shows direction, which a single number cannot.

    A flat series renders as a steady mid-level line, not dashes -- "unchanging"
    is a real reading and should not look like missing data.
    """
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return T.chrome + " " * width + T.off
    vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    if T.ascii:
        body = "-" * len(vals)
    elif hi - lo < 1e-9:
        body = SPARK[3] * len(vals)
    else:
        body = "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.99))] for v in vals)
    tone = T.chrome if hi - lo < 1e-9 else T.accent
    # left-pad so the sparkline's right edge stays anchored as it fills
    return " " * (width - len(body)) + tone + body + T.off


def status(frac, good_above=None, bad_above=None):
    """(tone, label). Label is mandatory -- color is never the only signal."""
    if frac is None:
        return T.chrome, "--"
    if good_above is not None:          # higher is better (cache hit rate)
        if frac >= good_above:
            return T.good, "OK"
        if frac >= good_above * 0.4:
            return T.warn, "LOW"
        return T.crit, "COLD"
    if bad_above is not None:           # higher is worse (KV pressure)
        if frac >= bad_above:
            return T.crit, "FULL"
        if frac >= bad_above * 0.8:
            return T.warn, "HIGH"
        return T.good, "OK"
    return T.chrome, ""


# ---------------------------------------------------------------- layout


def box(title, rows, width):
    """A titled panel. Borders are chrome: dim, so the data reads first."""
    if T.ascii:
        tl, tr, bl, br, h, v = "+", "+", "+", "+", "-", "|"
    else:
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    c = T.chrome
    head = "{}{}{} {} {}".format(c, tl, h, T.muted + title + c, h * max(0, width - vlen(title) - 5))
    out = [head + tr + T.off]
    for r in rows:
        out.append("{}{}{} {} {}{}{}".format(c, v, T.off, pad(r, width - 4), c, v, T.off))
    out.append(c + bl + h * (width - 2) + br + T.off)
    return out


def side_by_side(left, right, gap=2):
    lw = max((vlen(x) for x in left), default=0)
    out = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        out.append(pad(l, lw) + " " * gap + r)
    return out

HOP_BY_HOP = {"transfer-encoding", "content-length", "connection", "host",
              "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "upgrade"}

# Only these paths are generation requests worth measuring. Everything else
# (/tokenize, /slots, /health, /metrics) is proxied untouched and NOT counted.
# This matters: a driver that builds prompts by calling /tokenize in a loop
# would otherwise flood the rolling window with hundreds of empty records and
# drive the hit rate and request count to nonsense.
GEN_PATHS = ("/completion", "/completions", "/chat/completions", "/infill")


def is_generation(path):
    p = path.split("?", 1)[0].rstrip("/")
    return any(p.endswith(g) for g in GEN_PATHS)


# ---------------------------------------------------------------- stats


class Stats:
    """Rolling window of completed requests. Everything here is per-request."""

    def __init__(self, maxlen=500):
        self.lock = threading.Lock()
        self.reqs = deque(maxlen=maxlen)
        self.inflight = 0
        self.total = 0
        self.errors = 0

    def start(self):
        with self.lock:
            self.inflight += 1

    def finish(self, rec):
        with self.lock:
            self.inflight = max(0, self.inflight - 1)
            self.total += 1
            if rec:
                self.reqs.append(rec)
            else:
                self.errors += 1

    def snapshot(self, window_s=60.0):
        now = time.time()
        with self.lock:
            rows = [r for r in self.reqs if now - r["t"] <= window_s]
            inflight, total, errors = self.inflight, self.total, self.errors
        if not rows:
            return {"n": 0, "inflight": inflight, "total": total, "errors": errors}

        ttfts = sorted(r["ttft"] for r in rows if r.get("ttft"))
        itls = sorted(v for r in rows for v in r.get("itl", []))
        def cache_fraction(r):
            cached, fresh = r.get("cache_n"), r.get("prompt_n")
            if cached is None and fresh is None:
                return None
            total = (cached or 0) + (fresh or 0)
            return (cached or 0) / total if total else None

        # Cache state is the most useful way to split TTFT: a slow cold prompt
        # is expected, while a slow warm prompt points to scheduling or decode.
        warm_ttft = sorted(r["ttft"] for r in rows if r.get("ttft")
                           and (cache_fraction(r) is not None)
                           and cache_fraction(r) >= 0.7)
        cold_ttft = sorted(r["ttft"] for r in rows if r.get("ttft")
                           and (cache_fraction(r) is not None)
                           and cache_fraction(r) < 0.1)
        outputs = sorted(r["predicted_n"] for r in rows if r.get("predicted_n") is not None)
        stopped = [r for r in rows if r.get("stop_type") is not None]
        # The derived metric nothing else reports: of all prompt tokens seen,
        # what fraction did the server get to reuse instead of re-reading?
        reused = sum(r.get("cache_n") or 0 for r in rows)
        fresh = sum(r.get("prompt_n") or 0 for r in rows)
        hit = (reused / (reused + fresh)) if (reused + fresh) else None
        # Fraction of individual requests that got essentially no reuse.
        # Only requests that actually reported cache numbers can be judged --
        # counting unparsed ones as "cold" would invent a problem.
        judged = [r for r in rows
                  if r.get("cache_n") is not None or r.get("prompt_n") is not None]
        cold = sum(1 for r in judged
                   if (r.get("cache_n") or 0) < 0.1 * max(1, (r.get("prompt_n") or 0)
                                                          + (r.get("cache_n") or 0)))
        return {
            "n": len(rows), "inflight": inflight, "total": total, "errors": errors,
            "ttft": ttfts, "itl": itls, "hit": hit,
            "warm_ttft": warm_ttft, "cold_ttft": cold_ttft, "outputs": outputs,
            "limit_frac": (sum(r["stop_type"] == "limit" for r in stopped) / len(stopped))
                          if stopped else None,
            "cold_frac": (cold / len(judged)) if judged else None,
            "reused": reused, "fresh": fresh,
            "queue": sorted(r["queue_ms"] for r in rows if r.get("queue_ms") is not None),
            "prefill": sorted(r["prompt_ms"] for r in rows if r.get("prompt_ms") is not None),
            "rps": len(rows) / window_s,
            "recent": list(reversed(rows[-6:])),
        }


STATS = Stats()


def pct(vals, p):
    if not vals:
        return None
    k = min(len(vals) - 1, int(round(p / 100.0 * (len(vals) - 1))))
    return vals[k]


def latency_status(v, good, bad):
    """A label accompanies every coloured latency meter."""
    if v is None:
        return T.chrome, "--"
    if v <= good:
        return T.good, "OK"
    if v <= bad:
        return T.warn, "SLOW"
    return T.crit, "HIGH"


def latency_meter(v, scale, width, good, bad):
    tone, label = latency_status(v, good, bad)
    return meter((v / scale) if v is not None else None, width, tone), tone + label + T.off


# ---------------------------------------------------------------- proxy


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = ("127.0.0.1", 8080)

    def log_message(self, *a):
        pass

    def _dispatch(self):
        self._forward(self.command)

    # llama.cpp's HTTP surface isn't limited to GET/POST (CORS preflight uses
    # OPTIONS; slot/props management can use PUT or DELETE). Forward every
    # method the same way rather than silently 501-ing the rest.
    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _dispatch

    def _forward(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in HOP_BY_HOP}
        headers["Host"] = "{}:{}".format(*self.upstream)

        try:
            conn = http.client.HTTPConnection(*self.upstream, timeout=600)
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:
            self.send_error(502, "upstream unreachable: {}".format(e))
            return

        ctype = resp.getheader("Content-Type", "") or ""
        is_stream = "event-stream" in ctype

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        measured = is_generation(self.path)
        if is_stream:
            self._pump_stream(resp, measured)
        else:
            self._pump_plain(resp, measured)
        try:
            conn.close()
        except Exception:
            pass

    def _write_chunk(self, data):
        self.wfile.write(("%X\r\n" % len(data)).encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_chunks(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _pump_stream(self, resp, measured=True):
        """Forward SSE byte-for-byte while measuring it.

        The measurement must not change what it measures: every chunk is written
        downstream and flushed BEFORE it is parsed. Buffering here would inflate
        exactly the inter-token latency this tool exists to report.
        """
        if measured:
            STATS.start()
        t0 = time.perf_counter()
        ttft, marks, final = None, [], None
        try:
            for raw in resp:
                self._write_chunk(raw)             # forward first
                now = time.perf_counter()          # then measure
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if _content_of(obj):
                    if ttft is None:
                        ttft = (now - t0) * 1000.0
                    marks.append(now)
                if obj.get("stop") or "timings" in obj:
                    final = obj
            self._end_chunks()
        except Exception:
            if measured:
                STATS.finish(None)
            return
        if measured:
            STATS.finish(_record(ttft, marks, final, t0))

    def _pump_plain(self, resp, measured=True):
        if measured:
            STATS.start()
        t0 = time.perf_counter()
        try:
            data = resp.read()
            if data:
                self._write_chunk(data)
            self._end_chunks()
        except Exception:
            if measured:
                STATS.finish(None)
            return
        if not measured:
            return
        obj = None
        try:
            obj = json.loads(data.decode("utf-8", "replace"))
        except Exception:
            pass
        # A non-streamed generation still carries timings; anything else that
        # reached here is a generation call we could not parse, so count it but
        # contribute no latency samples.
        if isinstance(obj, dict) and "timings" in obj:
            STATS.finish(_record(None, [], obj, t0))
        else:
            STATS.finish({"t": time.time(), "ttft": None, "itl": [],
                          "prompt_n": None, "cache_n": None, "predicted_n": None,
                          "total_ms": (time.perf_counter() - t0) * 1000.0})


def _content_of(obj):
    """Token text from either the native or the OpenAI-compatible shape."""
    if obj.get("content"):
        return obj["content"]
    for ch in (obj.get("choices") or []):
        d = ch.get("delta") or {}
        if d.get("content"):
            return d["content"]
        if ch.get("text"):
            return ch["text"]
    return None


def _record(ttft, marks, final, t0):
    t = (final or {}).get("timings") or {}
    cache_n = t.get("cache_n")
    if cache_n is None:
        cache_n = (final or {}).get("tokens_cached")
    itl = [(marks[i] - marks[i - 1]) * 1000.0 for i in range(1, len(marks))]
    prompt_ms = t.get("prompt_ms")
    # TTFT decomposes into two very different problems: time spent WAITING to be
    # scheduled, and time spent READING the prompt. One is fixed by reducing
    # concurrency, the other by keeping the prompt prefix stable. A single TTFT
    # number cannot tell you which you have.
    queue_ms = None
    if ttft is not None and prompt_ms is not None:
        queue_ms = max(0.0, ttft - prompt_ms)
    return {
        "t": time.time(),
        "ttft": ttft,
        "itl": itl,
        "prompt_n": t.get("prompt_n"),
        "cache_n": cache_n,
        "predicted_n": t.get("predicted_n") or len(marks),
        "prompt_ms": prompt_ms,
        "queue_ms": queue_ms,
        "stop_type": (final or {}).get("stop_type")
                     or ("limit" if (final or {}).get("truncated") else None),
        "total_ms": (time.perf_counter() - t0) * 1000.0,
    }


class Threaded(ThreadingHTTPServer):
    # ThreadingHTTPServer already mixes in ThreadingMixIn; adding it again
    # breaks the MRO.
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- render


class History:
    """Short time series behind each headline number, for the sparklines."""

    def __init__(self, n=48):
        self.hit = deque(maxlen=n)
        self.kv = deque(maxlen=n)
        self.ttft = deque(maxlen=n)
        self.gap = deque(maxlen=n)

    def push(self, hit, kv, ttft, gap):
        self.hit.append(hit)
        self.kv.append(kv)
        self.ttft.append(ttft)
        self.gap.append(gap)


HIST = History()


def ms(v):
    """Latency with the unit dim, so the number carries the eye."""
    if v is None:
        return T.muted + "--" + T.off
    if v >= 1000:
        return "{}{:>6}{} {}s {}".format(T.ink, "%.2f" % (v / 1000.0), T.off, T.muted, T.off)
    return "{}{:>6}{} {}ms{}".format(T.ink, "%.0f" % v, T.off, T.muted, T.off)


def draw(lines):
    """Repaint in place. The dashboard's height varies with what the server
    exposes (GPU panel, slot rows, ...) and can exceed a modest terminal --
    writing past the last row forces the terminal to scroll, which permanently
    pushes the redraw's home position off screen and turns "in place" updates
    into a scrolling log. Cap to what actually fits instead.
    """
    # Only an interactive terminal can scroll out from under a redraw. When
    # stdout is redirected (a log file, the e2e test's captured pipe) there is
    # no real row count to respect, and shutil falls back to a fake 80x24-ish
    # default that would truncate a perfectly good log for no reason.
    if sys.stdout.isatty():
        rows = shutil.get_terminal_size((100, 40)).lines
        budget = max(1, rows - 1)
        if len(lines) > budget:
            hidden = len(lines) - (budget - 1)
            lines = lines[:budget - 1] + [
                T.chrome + "... {} more row(s) -- enlarge the terminal to see them".format(hidden) + T.off]
    out = [CSI + "H"]
    for ln in lines:
        out.append(ln + CSI + "K\n")
    out.append(CSI + "J")
    sys.stdout.write("".join(out))
    sys.stdout.flush()


def hero_cache(s, w):
    """The headline. Its job is a single number, so it gets a stat tile, not a
    chart -- with the meter and trend as supporting marks, not decoration."""
    hit = s.get("hit")
    tone, label = status(hit, good_above=0.7)
    rows = []
    if not s.get("n"):
        rows.append(T.muted + "waiting for traffic on port {}".format(s["_port"]) + T.off)
        rows.append(T.chrome + "point your client (or blindspots.py --host) here" + T.off)
        return box("PROMPT CACHE", rows, w)

    pctv = "{:>3.0f}%".format(hit * 100) if hit is not None else " --%"
    rows.append("{}  {}{:<4}{}  {}   {}".format(
        T.bold(pctv), tone, label, T.off,
        meter(hit, min(40, max(12, w - 40)), tone), spark(HIST.hit, 12)))
    cf = s.get("cold_frac")
    detail = "{} reused  {}  {} re-read".format(
        num(s.get("reused")), T.chrome + "·" + T.muted, num(s.get("fresh")))
    if cf is not None:
        detail += "   {}  {:.0%} of requests cold".format(
            T.chrome + "·" + T.muted, cf)
    rows.append(T.muted + detail + T.off)
    return box("PROMPT CACHE", rows, w)


def panel_latency(s, w):
    t, i = s.get("ttft") or [], s.get("itl") or []
    q, pf = s.get("queue") or [], s.get("prefill") or []
    name_w = 13
    bw = max(8, min(16, w - 37))

    def latency_row(name, value, scale, good, bad):
        bar, label = latency_meter(value, scale, bw, good, bad)
        return "{}{: <{}}{} {}  {}  {}".format(
            T.muted, name, name_w, T.off, ms(value), bar, label)

    rows = []
    rows.append(T.bold("TTFT") + T.muted + "  time to first token" + T.off)
    rows.append(latency_row("typical (p50)", pct(t, 50), 2000, 250, 1000))
    rows.append(latency_row("slow (p95)", pct(t, 95), 2000, 250, 1000))
    trend = " " + spark(HIST.ttft, 6)
    rows.append(latency_row("worst (p99)", pct(t, 99), 2000, 250, 1000) + trend)
    if q or pf:
        rows.append(T.chrome + "  first-token delay: waiting vs reading" + T.off)
        qm, pm = pct(q, 50), pct(pf, 50)
        rows.append(latency_row("wait for slot", qm, 2000, 100, 500))
        rows.append(latency_row("read prompt", pm, 2000, 250, 1000))
        if qm is not None and pm is not None and (qm + pm) > 0:
            dom = "waiting for a slot" if qm > pm else "reading the prompt"
            rows.append("{}  mostly {}{}".format(T.muted, dom, T.off))
    rows.append("")
    rows.append(T.bold("TOKEN GAP") + T.muted + "  between streamed chunks" + T.off)
    rows.append(latency_row("typical (p50)", pct(i, 50), 500, 50, 200))
    rows.append(latency_row("slow (p95)", pct(i, 95), 500, 50, 200))
    trend = " " + spark(HIST.gap, 6)
    rows.append(latency_row("worst (p99)", pct(i, 99), 500, 50, 200) + trend)

    p50i, p99i = pct(i, 50), pct(i, 99)
    if p50i and p99i and p99i > 4 * p50i:
        rows.append("")
        rows.append("{}STALL{}  p99 gap is {:.0f}x median".format(
            T.crit, T.off, p99i / p50i))
        rows.append(T.muted + "responses freezing mid-stream" + T.off)
    warm, cold = s.get("warm_ttft") or [], s.get("cold_ttft") or []
    if warm or cold:
        rows.append("")
        rows.append(T.bold("CACHE IMPACT") + T.muted + "  typical TTFT" + T.off)
        if warm:
            rows.append(latency_row("mostly cached ({})".format(len(warm)), pct(warm, 50),
                                    2000, 250, 1000))
        if cold:
            rows.append(latency_row("mostly new ({})".format(len(cold)), pct(cold, 50),
                                    2000, 250, 1000))
    if t and len(t) < 100:
        rows.append("")
        rows.append(T.chrome + "{} samples; p50=typical, p95/p99=slow tail".format(len(t)) + T.off)
    return box("LATENCY", rows, w)


def panel_kv(m, slots, w, metrics_ok=True):
    """Render only the cache state the particular server actually exposes.

    llama.cpp has shipped several /metrics and /slots schemas.  Some expose a
    server-wide KV capacity ratio; others (including current builds) expose
    cache-reuse counters and per-slot prompt progress but no capacity at all.
    Calling the latter "KV cache" makes a missing feature look like a fault,
    so fall back to a useful Slots & Queue panel instead.
    """
    kv = mget(m, "kv_cache_usage_ratio")
    kvt = mget(m, "kv_cache_tokens")
    rows = []

    if kv is not None:
        tone, label = status(kv, bad_above=0.9)
        rows.append("{}  {}{:<4}{} {}  {}".format(
            T.bold("{:>3.0f}%".format(kv * 100)), tone, label, T.off,
            meter(kv, max(8, w - 32), tone), spark(HIST.kv, 8)))
        if kvt is not None:
            rows.append(T.muted + "{} tokens resident".format(num(kvt)) + T.off)

    deferred = mget(m, "requests_deferred")
    busy_per = mget(m, "n_busy_slots_per_decode")
    if deferred is not None or busy_per is not None:
        rows.append("")
        bits = []
        if deferred is not None:
            tone = T.crit if deferred > 0 else T.muted
            bits.append("{}deferred{} {}{:.0f}{}".format(
                T.muted, T.off, tone, deferred, T.off))
        if busy_per is not None:
            bits.append("{}slots/decode{} {}{:.2f}{}".format(
                T.muted, T.off, T.ink, busy_per, T.off))
        rows.append("  ".join(bits))
        if deferred and deferred > 0:
            rows.append(T.muted + "requests are queueing, not running" + T.off)

    if slots:
        busy = sum(1 for x in slots if x.get("is_processing"))
        slot_frac = busy / len(slots)
        slot_tone, slot_label = status(slot_frac, bad_above=0.9)
        rows.append("")
        rows.append("{}SLOTS{}  {}{}{}{} of {} busy  {} {}{}{}".format(
            T.muted, T.off, T.ink, busy, T.off, T.muted, len(slots),
            meter(slot_frac, max(8, w - 35), slot_tone),
            slot_tone, slot_label, T.off))
        for sl in slots[:6]:
            # Old /slots schemas expose context occupancy; newer schemas
            # instead expose progress through the current prompt.  Render
            # whichever is present rather than filling a row with "--".
            np_, nc = sl.get("n_past"), sl.get("n_ctx")
            on = sl.get("is_processing")
            lead = "{}{:>2}{} {}{:<4}{}".format(
                T.chrome, sl.get("id", "?"), T.off,
                (T.accent if on else T.chrome), "busy" if on else "idle", T.off)
            if np_ is not None and nc:
                frac = np_ / nc
                rows.append("{} {}  {}{:>5}/{:<5}{}".format(
                    lead, meter(frac, max(6, w - 30), T.accent if on else T.chrome),
                    T.muted, num(np_), num(nc), T.off))
                continue
            prompt = sl.get("n_prompt_tokens")
            cached = sl.get("n_prompt_tokens_cache")
            processed = sl.get("n_prompt_tokens_processed")
            parts = []
            if prompt is not None:
                parts.append("prompt {}".format(num(prompt)))
            if cached is not None:
                parts.append("cached {}".format(num(cached)))
            if processed is not None:
                parts.append("read {}".format(num(processed)))
            if prompt and cached is not None and 0 <= cached <= prompt:
                reuse = cached / prompt
                tone, label = status(reuse, good_above=0.7)
                reuse_row = "{}  reuse {} {}{}".format(
                    lead, meter(reuse, max(6, w - 22), tone), tone, label + T.off)
                # Two compact rows are more legible than squeezing prompt
                # counts and a meter into one narrow side-by-side panel.
                rows.append(lead + ("  " + T.muted + "  ".join(parts) + T.off if parts else ""))
                rows.append(reuse_row)
            else:
                rows.append(lead + ("  " + T.muted + "  ".join(parts) + T.off if parts else ""))

    # Never imply a capacity reading where the server exposes none.
    return box("KV CACHE" if kv is not None else "SLOTS & QUEUE", rows, w)


def panel_server(s, gpu, w, m=None, slots=None):
    rows = []
    # Every bar in this panel shares a ruler.  Different rulers make adjacent
    # values look comparable when they are not.
    bar_w = max(8, min(28, w - 44))
    in_flight = s.get("inflight", 0)
    rows.append(T.bold("REQUEST LOAD"))
    if slots:
        capacity = len(slots)
        load = min(1.0, in_flight / capacity)
        tone, label = status(load, bad_above=0.9)
        rows.append("{}active requests{}  {}{} / {}{}  {} {}{}{}".format(
            T.muted, T.off, T.ink, in_flight, capacity, T.off,
            meter(load, bar_w, tone), tone, label, T.off))
    else:
        rows.append("{}active requests{}  {}{}{}".format(
            T.muted, T.off, T.ink, in_flight, T.off))
    rows.append("{}completed through proxy{}  {}{}{}     {}proxy errors{}  {}{}{}     "
                "{}traffic{}  {:.2f} req/s".format(
                    T.muted, T.off, T.ink, num(s.get("total", 0)), T.off,
                    T.muted, T.off, T.crit if s.get("errors") else T.ink,
                    s.get("errors", 0), T.off, T.muted, T.off, s.get("rps", 0.0)))
    if m is not None:
        pre, pre_stale = RATES.update(m, "prompt_tokens_total", "prompt_seconds_total")
        dec, dec_stale = RATES.update(m, "tokens_predicted_total",
                                      "tokens_predicted_seconds_total")
        if pre is not None or dec is not None:
            mark = " {}(last active speed){}".format(T.chrome, T.off) \
                if (pre_stale or dec_stale) else \
                " {}(currently processing){}".format(T.chrome, T.off)
            rows.append("")
            rows.append(T.bold("MODEL SPEED"))
            rows.append("{}read prompt{}  {}{}{} tok/s     {}generate text{}  {}{}{} tok/s{}".format(
                T.muted, T.off, T.ink, num(pre), T.off,
                T.muted, T.off, T.ink, num(dec), T.off, mark))
    outputs = s.get("outputs") or []
    limit_frac = s.get("limit_frac")
    if outputs or limit_frac is not None:
        rows.append("")
        rows.append(T.bold("RESPONSE SIZE"))
    if outputs:
        p50, p95 = pct(outputs, 50), pct(outputs, 95)
        # Length is workload information, not an error state.  The 256-token
        # ruler makes it scannable without calling a long answer "bad".
        rows.append("{}typical{}  {}{}{} tokens".format(
            T.muted, T.off, T.ink, num(p50), T.off))
        rows.append("{}long (p95){}  {}{}{} tokens  {}{} / 256{}".format(
            T.muted, T.off, T.ink, num(p95), T.off,
            meter(p95 / 256.0, bar_w, T.accent), T.muted, T.off))
    if limit_frac is not None:
        tone, label = status(limit_frac, bad_above=0.1)
        rows.append("{}hit requested token limit{}  {:>3.0f}%  {} {}{}{}".format(
            T.muted, T.off, limit_frac * 100,
            meter(limit_frac, bar_w, tone), tone, label, T.off))
    if gpu:
        rows.append("")
        rows.append(T.bold("GPU"))
        util_tone, util_label = status(gpu["util"] / 100.0, bad_above=0.95)
        mem_frac = gpu["used"] / gpu["total"] if gpu["total"] else None
        mem_tone, mem_label = status(mem_frac, bad_above=0.9)
        if gpu["temp"] >= 85:
            temp_tone, temp_label = T.crit, "HOT"
        elif gpu["temp"] >= 75:
            temp_tone, temp_label = T.warn, "WARM"
        else:
            temp_tone, temp_label = T.good, "OK"
        # The memory reading is naturally wider than a percentage or a
        # temperature.  Reserve one value column before the ruler so it does
        # not make its otherwise identical bar look longer by starting later.
        gpu_rows = (
            ("compute use", "{:>3.0f}%".format(gpu["util"]),
             gpu["util"] / 100.0, util_tone, util_label),
            ("memory used", "{} / {} MiB".format(num(gpu["used"]), num(gpu["total"])),
             mem_frac, mem_tone, mem_label),
            ("temperature", "{:.0f}C".format(gpu["temp"]),
             gpu["temp"] / 100.0, temp_tone, temp_label),
        )
        label_w = max(len(label) for label, _, _, _, _ in gpu_rows)
        value_w = max(len(value) for _, value, _, _, _ in gpu_rows)
        for label, value, fraction, tone, state in gpu_rows:
            rows.append("{}{}{}  {}  {} {}{}{}".format(
                T.muted, label.ljust(label_w), T.off, value.rjust(value_w),
                meter(fraction, bar_w, tone), tone, state, T.off))
    return box("SERVER", rows, w)


def render(args, m, slots, gpu, s):
    s["_port"] = args.listen
    total = max(64, min(shutil.get_terminal_size((100, 40)).columns - 1, 120))

    # feed the sparklines
    t, i = s.get("ttft") or [], s.get("itl") or []
    HIST.push(s.get("hit"), mget(m, "kv_cache_usage_ratio"), pct(t, 99), pct(i, 99))

    hdr_warn = ""
    if not s.get("_metrics_ok", True):
        hdr_warn = "  " + T.crit + "/metrics down" + T.off
    if slots is None:
        hdr_warn += "  " + T.warn + "/slots off" + T.off
    head = "{}   {}proxy :{} → :{}{}{}".format(
        T.bold("llmwatch"), T.chrome, args.listen, args.upstream_port, T.off, hdr_warn)
    clock = "{}{}  {}last {}s  n={}{}".format(
        T.muted, time.strftime("%H:%M:%S"), T.chrome, args.window, s.get("n", 0), T.off)
    L = [pad(head, total - vlen(clock)) + clock, ""]

    L += hero_cache(s, total)
    L.append("")

    ok = s.get("_metrics_ok", True)
    # The latency panel carries explanations as well as numbers.  A half-width
    # box is only comfortable on a genuinely wide terminal; below that, stack
    # panels so labels stay readable instead of being clipped into shorthand.
    if total >= 112:
        half = (total - 2) // 2
        L += side_by_side(panel_latency(s, half), panel_kv(m, slots, half, ok))
    else:
        L += panel_latency(s, total)
        L.append("")
        L += panel_kv(m, slots, total, ok)
    L.append("")
    L += panel_server(s, gpu, total, m, slots)
    L.append("")
    L.append(T.chrome + "ctrl-c to quit" + T.off)
    return L


# ---------------------------------------------------------------- main


def main():
    args = build_parser(__doc__).parse_args()

    global T
    T = Theme(ascii_only=args.ascii, force_mono=args.no_color)

    base = "http://{}:{}".format(args.upstream_host, args.upstream_port)
    ProxyHandler.upstream = (args.upstream_host, args.upstream_port)

    if args.debug:
        raw = http_get(base + "/metrics")
        print("GET {}/metrics".format(base))
        if raw is None:
            print("  UNREACHABLE -- the server was started without --metrics")
        else:
            met = parse_prom(raw)
            print("  {} metric(s):".format(len(met)))
            for k in sorted(met):
                print("    {:<46} {}".format(k, met[k]))
            print("  kv_cache_usage_ratio resolves to:",
                  mget(met, "kv_cache_usage_ratio"))
        raw = http_get(base + "/slots")
        print("\nGET {}/slots".format(base))
        if raw is None:
            print("  UNREACHABLE -- the server was started without --slots")
        else:
            try:
                sl = json.loads(raw)
                sl = sl if isinstance(sl, list) else sl.get("slots", [])
                print("  {} slot(s); keys on slot 0:".format(len(sl)))
                if sl:
                    for k in sorted(sl[0]):
                        print("    " + k)
            except json.JSONDecodeError:
                print("  unparseable: " + raw[:200])
        return 0

    srv = None
    if not args.no_proxy:
        try:
            srv = Threaded((args.listen_host, args.listen), ProxyHandler)
        except OSError as e:
            print("cannot bind {}:{}: {}".format(args.listen_host, args.listen, e))
            return 1
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    sys.stdout.write(CSI + "2J" + CSI + "?25l")
    try:
        while True:
            raw_m = http_get(base + "/metrics")
            m = parse_prom(raw_m)
            slots = None
            raw = http_get(base + "/slots")
            if raw:
                try:
                    p = json.loads(raw)
                    slots = p if isinstance(p, list) else p.get("slots")
                except json.JSONDecodeError:
                    pass
            snap = STATS.snapshot(args.window)
            snap["_metrics_ok"] = raw_m is not None
            draw(render(args, m, slots, get_gpu(), snap))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(CSI + "?25h\n")
        if srv:
            srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
