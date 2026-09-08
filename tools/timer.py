"""Pipeline step timer — records elapsed time per step to a JSON file.

Usage (context manager):
    timer = StepTimer("/path/to/timing.json")
    with timer.record("1d_filter_interior"):
        do_work()

Usage (decorator):
    timer = StepTimer("/path/to/timing.json")

    @timer.record("1d_filter_interior")
    def main():
        ...

The JSON file is created/updated after each step and can be read mid-run.
Existing entries are preserved (safe to rerun skipped steps).

JSON format:
    {
      "1d_filter_interior": {
        "start":       "2026-06-12T01:23:45",
        "elapsed_sec": 127.4,
        "elapsed_min": 2.12,
        "human":       "2m 7.4s"
      },
      ...
    }

If output_path is None the timer still prints but does not write to disk.
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class _Step:
    """Returned by StepTimer.record(). Works as context manager or decorator."""

    def __init__(self, timer: "StepTimer", name: str) -> None:
        self._timer = timer
        self._name  = name

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "_Step":
        self._t0       = time.perf_counter()
        self._start_ts = datetime.now().isoformat(timespec="seconds")
        return self

    def __exit__(self, *_) -> None:
        self._timer._commit(self._name, self._start_ts,
                            time.perf_counter() - self._t0)

    # ── decorator ────────────────────────────────────────────────────────────

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _Step(self._timer, self._name):
                return fn(*args, **kwargs)
        return wrapper


class StepTimer:
    """Per-dataset pipeline timer. Thread-safe writes via atomic replace."""

    def __init__(self, output_path: Optional[Union[str, Path]]) -> None:
        self._path: Optional[Path] = Path(output_path) if output_path else None
        self._data: dict = {}
        if self._path and self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    # ── public API ────────────────────────────────────────────────────────────

    def record(self, name: str) -> _Step:
        """Return a _Step that can be used as a context manager or decorator."""
        return _Step(self, name)

    # ── internal ──────────────────────────────────────────────────────────────

    def _commit(self, name: str, start_ts: str, elapsed_sec: float) -> None:
        human = _fmt(elapsed_sec)
        self._data[name] = {
            "start":       start_ts,
            "elapsed_sec": round(elapsed_sec, 1),
            "elapsed_min": round(elapsed_sec / 60, 2),
            "human":       human,
        }
        print(f"[timer] {name}: {human}")
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
            tmp.replace(self._path)


def _fmt(sec: float) -> str:
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    if m:
        return f"{int(m)}m {s:.1f}s"
    return f"{s:.1f}s"


# ── convenience: import path helper used by scripts ──────────────────────────

def _repo_root() -> Path:
    """Return repo root (parent of this file's parent)."""
    return Path(__file__).parent.parent
