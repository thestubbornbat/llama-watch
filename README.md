# llama-watch

`llama-watch` is a terminal dashboard and transparent reverse proxy for a
local [llama.cpp](https://github.com/ggml-org/llama.cpp) server. It shows
request-level prompt-cache reuse, time-to-first-token, inter-token stalls,
response size, slot state, server speed, and NVIDIA GPU state.

It fills in measurements that `/metrics` alone cannot provide: cache reuse and
latency are measured from requests as they pass through the proxy.

## Requirements

- Python 3.9 or newer
- A running llama.cpp `llama-server` with `--metrics`; `--slots` is strongly
  recommended.
- No Python runtime dependencies.
- `nvidia-smi` on `PATH` is optional and enables the GPU panel.

## Install

From a checkout, for an isolated command-line install:

```bash
pipx install .
```

For development from a checkout:

```bash
python -m pip install -e .
```

## Quick start

Start llama.cpp on loopback, with metrics and slots enabled:

```bash
llama-server -m model.gguf --metrics --slots -np 4 -c 8192 --host 127.0.0.1 --port 8080
```

In a second terminal, start the dashboard and proxy:

```bash
llama-watch --upstream-host 127.0.0.1 --upstream-port 8080 --listen 8081
```

Point the client that normally calls `http://127.0.0.1:8080` to
`http://127.0.0.1:8081` instead. Requests are forwarded unchanged; generation
requests are observed as their response streams through.

## Proxy and security model

Both the upstream and proxy listener default to `127.0.0.1`. This is the safe
configuration for a local model server: llama-watch does not implement
authentication, TLS, or access control.

To make the proxy reachable from another machine, put an authenticated TLS
reverse proxy in front of it, or explicitly choose a bind address only on a
trusted private network:

```bash
llama-watch --listen-host 192.168.1.20 --listen 8081
```

The proxy preserves streaming responses and uses HTTP/1.1 chunked transfer
encoding downstream. It removes hop-by-hop headers and generates its own
transfer framing, avoiding conflicting upstream `Content-Length` headers.

## Useful commands

```bash
# Inspect exactly what this llama.cpp build exposes.
llama-watch --debug

# Run only the display/collectors; do not accept client requests.
llama-watch --no-proxy

# Friendly output for basic terminals or logs.
llama-watch --ascii --no-color
```

Use `llama-watch --help` for all options. From an uninstalled checkout, run
`PYTHONPATH=src python -m llamawatch`.

## Development

```bash
python -m unittest discover -s tests -v
```

The included end-to-end test starts a mock llama.cpp server, sends both
streaming and non-streaming requests through the proxy, and verifies that the
dashboard renders request metrics.
