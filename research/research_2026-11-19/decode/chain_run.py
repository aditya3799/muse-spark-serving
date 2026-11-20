"""Execute the preserved four-block decoder benchmark with exclusive GPU ownership."""
import argparse
import fcntl
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "research_2026-11-18_rl_prefill/decoder_chain"
sys.path.insert(0, str(ROOT / "research_2026-11-18_rl"))
from run_pair import telemetry

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--cases", nargs="+", default=["1x74160", "4x8192", "16x4096"])
p.add_argument("--rounds", type=int, default=8)
args = p.parse_args()
directory = HERE / f"chain_{args.seed}"
directory.mkdir(exist_ok=False)
sys.path.insert(0, str(OLD))
sys.argv = [str(OLD / "bench.py"), "--seed", str(args.seed), "--cases", *args.cases,
            "--rounds", str(args.rounds), "--out", str(directory / "result.json")]
with (ROOT / "research_2026-11-18_rl/.gpu.lock").open("a+") as guard:
    fcntl.flock(guard, fcntl.LOCK_EX)
    with telemetry(directory):
        runpy.run_path(str(OLD / "bench.py"), run_name="__main__")
