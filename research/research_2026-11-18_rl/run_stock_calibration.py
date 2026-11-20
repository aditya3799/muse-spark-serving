"""Measure frozen stock and release on the large verify and captured prefill shapes."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from run_pair import telemetry

here=Path(__file__).resolve().parent
directory=here.parent/"research_2026-11-18_rl_prefill/calibration"
cuda=Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
headers=Path(sys.prefix)/"python-headers/usr/include"
env=dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS="1",
         PATH=f"{Path(sys.prefix)/'bin'}:{cuda/'bin'}:{os.environ.get('PATH','')}",
         CPATH=f"{headers/'python3.12'}:{headers}:{os.environ.get('CPATH','')}",
         LD_LIBRARY_PATH=f"{cuda/'lib'}:/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH','')}")
suffix=""
if any((directory/name/"calibration.json").exists() for name in ("wide_retry_pilot","wide_retry_holdout")):
    suffix=datetime.now(timezone.utc).strftime("_%Y%m%dT%H%M%S%fZ")
for stage,seed in [("wide_retry_pilot",35791),("wide_retry_holdout",94063)]:
    out=directory/(stage+suffix)
    out.mkdir(exist_ok=True)
    with telemetry(out):
        subprocess.run([sys.executable,"-B","-u",str(directory/"paired.py"),"--seed",str(seed),
            "--rounds","8","--rows","256","768","3072","--out",str(out)],env=env,check=True)
    print(stage,"complete",flush=True)
subprocess.run([sys.executable,"-B",str(here/"calibration_summary.py")],check=True)