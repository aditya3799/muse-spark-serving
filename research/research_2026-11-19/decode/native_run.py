"""Run a preserved native-stack experiment with isolated receipts and telemetry."""
import argparse
import fcntl
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "research_2026-11-18_rl/swa_schedule"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(OLD.parent))
from run_pair import telemetry

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--kind", choices=["native", "window"], default="native")
p.add_argument("--smoke", action="store_true")
args = p.parse_args()
out = HERE / f"{args.kind}_{args.seed}"
out.mkdir(exist_ok=False)
base = ROOT / "release_2026-11-18/candidate.diff"
assert hashlib.sha256(base.read_bytes()).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
source = (OLD / "native_stack.py").read_text()
old_file = 'file=here/(("native_smoke_" if args.smoke else "native_stack_")+str(args.seed)+".json")'
assert source.count(old_file) == 1
source = source.replace(old_file, 'file=Path(' + repr(str(out / "result.json")) + ')')
if args.kind == "window":
    trial = (OLD / "window_trials.py").read_text()
    start = trial.index('def replace(old,new):')
    end = trial.index('exec(compile(source,')
    transformations = trial[start:end]
    transformations = transformations.replace('replace(\'("native_smoke_" if args.smoke else "native_stack_")\',\'("window_smoke_" if args.smoke else "window_stack_")\')', '')
    context = dict(source=source, hashlib=hashlib)
    exec(transformations, context)
    source = context["source"]
sys.argv = [str(OLD / "native_stack.py"), "--seed", str(args.seed)]
if args.smoke:
    sys.argv.append("--smoke")
(out / "executed.py").write_text(source)
(out / "manifest.json").write_text(json.dumps(dict(seed=args.seed, kind=args.kind, started=time.time(),
    native_source_sha256=hashlib.sha256((OLD / "native_stack.py").read_bytes()).hexdigest(),
    executed_source_sha256=hashlib.sha256(source.encode()).hexdigest()), indent=2))
with (OLD.parent / ".gpu.lock").open("a+") as guard:
    fcntl.flock(guard, fcntl.LOCK_EX)
    with telemetry(out):
        exec(compile(source, str(OLD / "native_stack.py"), "exec"), {"__file__": str(OLD / "native_stack.py"), "__name__": "__main__"})
