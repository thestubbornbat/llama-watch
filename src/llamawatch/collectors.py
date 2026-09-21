"""Collectors for llama.cpp Prometheus metrics and NVIDIA GPU counters."""

import shutil
import subprocess
import urllib.request


def http_get(url, timeout=1.5):
    """Return a UTF-8 HTTP response, or ``None`` when the endpoint is down."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def parse_prom(text):
    """Parse the scalar subset of Prometheus exposition used by llama.cpp."""
    metrics = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            metrics[parts[0].split("{", 1)[0].strip()] = float(parts[1])
        except ValueError:
            pass
    return metrics


def mget(metrics, *suffixes):
    """Find a metric despite llama.cpp's changing metric-name prefixes."""
    for suffix in suffixes:
        if suffix in metrics:
            return metrics[suffix]
    for suffix in suffixes:
        for key, value in metrics.items():
            if key.endswith(suffix) or key.endswith(":" + suffix):
                return value
    return None


class Rate:
    """Interval rate from llama.cpp token and active-seconds counters."""

    def __init__(self):
        self.prev = {}
        self.last = {}

    def update(self, metrics, token_key, seconds_key):
        tokens = mget(metrics, token_key)
        seconds = mget(metrics, seconds_key)
        if tokens is None or seconds is None:
            return None, False
        previous = self.prev.get(token_key)
        self.prev[token_key] = (tokens, seconds)
        if previous is not None:
            delta_tokens, delta_seconds = tokens - previous[0], seconds - previous[1]
            if delta_seconds > 1e-9 and delta_tokens >= 0:
                self.last[token_key] = delta_tokens / delta_seconds
                return self.last[token_key], False
        held = self.last.get(token_key)
        if held is None and seconds and seconds > 1e-9:
            held = tokens / seconds
        return held, held is not None


RATES = Rate()


def get_gpu():
    """Read the first NVIDIA GPU, or return ``None`` when unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        output = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        ).stdout.strip().splitlines()
        fields = [value.strip() for value in output[0].split(",")]
        return {"util": float(fields[0]), "used": float(fields[1]),
                "total": float(fields[2]), "temp": float(fields[3])}
    except Exception:
        return None
