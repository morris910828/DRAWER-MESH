#!/usr/bin/env python
"""CLI helper: append a single timing entry to a JSON file.

Called from shell scripts to record elapsed time of non-Python steps
(e.g. nerfstudio train.py, extract_mesh.py, texture.py).

Usage:
    python tools/record_timing.py \\
        --out   <timing.json> \\
        --name  step_1b_sdf_train \\
        --start "2026-06-12T01:23:45" \\
        --sec   6120
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from timer import _fmt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",   required=True, help="Timing JSON file path")
    ap.add_argument("--name",  required=True, help="Step name / key")
    ap.add_argument("--start", required=True, help="ISO-8601 start timestamp")
    ap.add_argument("--sec",   required=True, type=float, help="Elapsed seconds")
    args = ap.parse_args()

    path = Path(args.out)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}

    data[args.name] = {
        "start":       args.start,
        "elapsed_sec": round(args.sec, 1),
        "elapsed_min": round(args.sec / 60, 2),
        "human":       _fmt(args.sec),
    }
    print(f"[timer] {args.name}: {_fmt(args.sec)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


if __name__ == "__main__":
    main()
