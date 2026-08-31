"""
logging.py — Tiny stdout/stderr tee so training terminal output also lands in a file.

Usage:
    from dreamer.utils.logging import tee_stdout
    log_file = tee_stdout('logs/pendulum_20260817-120000.log')
    print('hello')     # goes to terminal AND to file
    log_file.close()   # at shutdown

Design: transparent to existing `print(..., flush=True)` calls in train.py.
"""

import os
import sys


class _Tee:
    """Writes to multiple streams (typically the real stdout and a log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def tee_stdout(path: str):
    """
    Redirects sys.stdout and sys.stderr through a Tee that also writes to `path`.
    Creates the parent directory if needed. Opens the file line-buffered so tail -f works.

    Args:
        path: log file path (parent dir will be created).
    Returns:
        The opened file handle. Caller should close it at shutdown.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    f = open(path, 'w', buffering=1)   # line-buffered
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f
