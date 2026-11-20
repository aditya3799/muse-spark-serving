"""Run any local experiment adapter twice and save its paired evidence and reward."""
import argparse
import fcntl
from datetime import datetime, timezone
import tempfile
import hashlib
import json
from contextlib import contextmanager
import shutil
from pathlib import Path
import subprocess
import sys

from score_trials import BASE, ROOT

HERE = Path(__file__).resolve().parent


@contextmanager
def telemetry(directory):
    binary = shutil.which("nvidia-smi")
    if not binary and Path("/usr/lib/wsl/lib/nvidia-smi").exists():
        binary = "/usr/lib/wsl/lib/nvidia-smi"
    if not binary:
        yield
        return
    fields="timestamp,name,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used"
    with (directory/"gpu_telemetry.log").open("w") as log:
        proc=subprocess.Popen([binary,"--query-gpu="+fields,"--format=csv","-l","1"],stdout=log,stderr=log)
        try:
            yield
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name",required=True)
    p.add_argument("--seeds",required=True,type=int,nargs=2)
    p.add_argument("command",nargs=argparse.REMAINDER)
    args=p.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        p.error("name accepts letters, digits, hyphens and underscores")
    if len(set(args.seeds))!=2:
        p.error("choose two distinct input seeds")
    cmd=args.command[1:] if args.command[:1]==["--"] else args.command
    if not cmd or not all(any(token in v for v in cmd) for token in ("{seed}","{output}")):
        p.error("provide the adapter command with both {seed} and {output} arguments")
    assert hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest()==BASE
    outdir=HERE/"runs"/args.name
    outdir.parent.mkdir(parents=True,exist_ok=True)
    try:
        outdir.mkdir()
    except FileExistsError:
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir=Path(tempfile.mkdtemp(prefix=args.name+"-"+stamp+"-",dir=outdir.parent))
    print("Benchmark receipts:",outdir,flush=True)
    paths=[]
    with (HERE/".gpu.lock").open("a+") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        with telemetry(outdir):
            for phase,seed in zip(("pilot","holdout"),args.seeds):
                path=outdir/f"{phase}.json"
                expanded=[v.replace("{seed}",str(seed)).replace("{output}",str(path)) for v in cmd]
                subprocess.run(expanded,cwd=ROOT,check=True)
                data=json.loads(path.read_text())
                if data["baseline_sha256"]!=BASE or data["seed"]!=seed or data.get("complete") is False:
                    raise ValueError(f"{phase}: expected complete measurements with the specified seed and parent")
                paths.append(path)
                if data.get("correct") is not True or data.get("active") is not True:
                    break
    score_cmd=[sys.executable,str(HERE/"score_trials.py"),str(paths[0])]
    if len(paths)==2:
        score_cmd.extend(["--holdout",str(paths[1])])
    subprocess.run(score_cmd+["--output",str(outdir/"reward.json")],check=True)
    subprocess.run([sys.executable,str(HERE/"ledger.py")],check=True)


if __name__=="__main__":main()
