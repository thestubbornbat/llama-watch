#!/usr/bin/env python3
"""Compatibility launcher for llama-watch.

Install the package and use ``llama-watch`` (or ``python -m llamawatch``)
for new deployments.
"""

import os
import sys


# Let ``python llmwatch.py`` continue to work from an uninstalled checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llamawatch.app import main


if __name__ == "__main__":
    raise SystemExit(main())
