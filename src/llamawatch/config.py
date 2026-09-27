"""Command-line configuration for llama-watch."""

import argparse


def build_parser(description):
    """Create the public CLI parser without importing monitor internals."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--listen-host", default="127.0.0.1",
                        help="proxy bind address (default: loopback only)")
    parser.add_argument("--listen", type=int, default=8081, help="proxy port")
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, default=8080)
    parser.add_argument("--window", type=int, default=60, help="rolling window seconds")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="append each measured request as a JSON line to PATH")
    parser.add_argument("--no-proxy", action="store_true",
                        help="render only; do not start the proxy")
    parser.add_argument("--ascii", action="store_true",
                        help="ASCII marks only, for terminals without block glyphs")
    parser.add_argument("--no-color", action="store_true", help="monochrome output")
    parser.add_argument("--debug", action="store_true",
                        help="print exactly what the server exposes, then exit")
    return parser
