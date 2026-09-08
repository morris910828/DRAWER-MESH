#!/usr/bin/env python3
"""Parse an experiment's tensorboard event file -> append best eval-PSNR row to a CSV.
Usage: _parse_best.py <exp_out_dir> <name> <results_csv>"""
import sys, glob, os
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util

out_dir, name, res = sys.argv[1], sys.argv[2], sys.argv[3]
fs = glob.glob(os.path.join(out_dir, "events.out.tfevents*"))
psnr, ssim = [], []
if fs:
    f = max(fs, key=os.path.getsize)
    for ev in event_file_loader.EventFileLoader(f).Load():
        if not ev.summary:
            continue
        for v in ev.summary.value:
            t = v.tag.lower()
            if t in ("eval images metrics/psnr", "eval images metrics/ssim"):
                try:
                    x = float(tensor_util.make_ndarray(v.tensor))
                except Exception:
                    x = v.simple_value
                (psnr if "psnr" in t else ssim).append((ev.step, x))

def fmt(rows):
    return sorted({(s, x) for s, x in rows if x == x})

psnr, ssim = fmt(psnr), fmt(ssim)
if psnr:
    bstep, bpsnr = max(psnr, key=lambda r: r[1])
    lstep, lpsnr = psnr[-1]
    bssim = next((x for s, x in ssim if s == bstep), float("nan"))
    verdict = "PROMISING(best>60k)" if bstep > 60000 else "early-peak(<60k)"
    row = f"{name},{bpsnr:.3f},{bstep},{bssim:.3f},{lpsnr:.3f},{lstep},{verdict}"
else:
    row = f"{name},NA,NA,NA,NA,NA,no-eval-data"
with open(res, "a") as fh:
    fh.write(row + "\n")
print("RESULT:", row)
