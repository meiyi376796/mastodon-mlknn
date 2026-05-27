"""Technical logger with timestamp, level, module, and content.

Format: HH:MM:SS  LVL  module    message
Levels: INF (info), WRN (warn), ERR (error), SKP (skip), OK (success)
Uses low-saturation ANSI colors for level differentiation.
"""

import sys
from datetime import datetime


_DIM    = "\033[2m"
_RESET  = "\033[0m"
_RED    = "\033[2;31m"
_YELLOW = "\033[2;33m"
_CYAN   = "\033[2;36m"
_GREEN  = "\033[2;32m"
_MAGENTA = "\033[2;35m"


class Logger:
    """Per-module logger emitting timestamped, level-tagged lines to stderr."""

    def __init__(self, module: str, stream=sys.stderr):
        self._module = module
        self._stream = stream

    def _emit(self, level: str, color: str, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        label = f"{color}{level}{_RESET}"
        line = f"{_DIM}{ts}{_RESET}  {label}  {_DIM}{self._module:<8}{_RESET}  {msg}"
        print(line, file=self._stream)

    def info(self, msg: str):
        self._emit("INF", _CYAN, msg)

    def warn(self, msg: str):
        self._emit("WRN", _YELLOW, msg)

    def error(self, msg: str):
        self._emit("ERR", _RED, msg)

    def skip(self, msg: str):
        self._emit("SKP", _MAGENTA, msg)

    def ok(self, msg: str):
        self._emit(" ✓ ", _GREEN, msg)

    def section(self, name: str):
        rule = "─" * 40
        print(f"\n{_DIM}{rule}{_RESET}", file=self._stream)
        print(f"  {name}", file=self._stream)

    def header(self, name: str = "mastodon-mlknn"):
        rule = "─" * 40
        print(f"\n{_DIM}{rule}{_RESET}", file=self._stream)
        print(f"  {name}", file=self._stream)

    def blank(self):
        print(file=self._stream)
