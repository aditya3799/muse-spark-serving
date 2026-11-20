"""Scope a single local GPU run and retain its last progress in case of a stall."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
here=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--out",required=True,type=Path)
p.add_argument("--seed",type=int,required=True)
p.add_argument("--rows",type=int,nargs="+",required=True)
p.add_argument("--variant",required=True)
args=p.parse_args()
assert not args.out.exists(), "supply a new receipt"
launch=[sys.executable,str(here/"launch_hold.py"),"paired_hold","--seed",str(args.seed),"--rows",*map(str,args.rows),"--variant",args.variant,"--out",str(args.out)]
started=time.monotonic()
with subprocess.Popen(launch,start_new_session=True) as child:
    last=None
    since=started
    while child.poll() is None:
        progress=args.out.stat().st_mtime_ns if args.out.exists() else None
        if progress!=last:
            last=progress
            since=time.monotonic()
        threshold=180 if progress is None else 65
        if time.monotonic()-since>threshold:
            print(f"Stopped own experiment after {threshold} seconds without a completed saved round.",flush=True)
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
            sys.exit(94)
        time.sleep(1)
    sys.exit(child.returncode)
