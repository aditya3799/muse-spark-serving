"""Allocate the next local GPU trial from signed rewards on a shared operation."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARM_DIR = ROOT / "research_2026-11-18_rl_prefill"
sys.path.insert(0, str(ARM_DIR))


def child(group, seed, out):
    import paired
    from score_trials import BASE
    import torch
    assert hashlib.sha256((ROOT/"release_2026-11-18/candidate.diff").read_bytes()).hexdigest() == BASE
    native = paired.proposed
    args = native.make_cpp_args(torch.bfloat16, native.is_arch_support_pdl())
    module = paired.original_load(
        f"muse_route_scaled_activation_group_{group}", *args,
        cuda_files=[str(ARM_DIR/"route_activation.cuh")],
        extra_cuda_cflags=native.get_activation_cuda_cflags(),
        extra_include_paths=[str(native.KERNEL_PATH/"csrc")],
        cuda_wrappers=[("run", f"RouteActivationKernel<{args}, {group}>::run")])
    calls = []
    class Counter:
        def run(self, *tensors):
            calls.append(tensors[0].shape[0])
            module.run(*tensors)
    native.route_activation_module = lambda: Counter()
    result = dict(agent="prefill-search",candidate=f"ballot_group_{group}",
                  scope="native scaled W13-SiLU boundary, three prefill shapes",seed=seed,
                  baseline_sha256=BASE, correct=False,active=False,complete=False,
                  gpu=torch.cuda.get_device_name(), bench={"instances":1,"shapes":{}})
    for m, pattern in [(753,"uniform"),(3012,"clustered"),(16384,"uniform")]:
        result["bench"]["shapes"][f"M{m}_{pattern}"] = paired.execute(m,pattern,seed,12)
        torch.cuda.empty_cache()
    result.update(correct=True,active=all(8*m in calls for m in (753,3012,16384)),complete=True)
    result["variant_identity"] = dict(header_sha256=hashlib.sha256((ARM_DIR/"route_activation.cuh").read_bytes()).hexdigest(), group=group)
    out.write_text(json.dumps(result,indent=2)+"\n")


def search():
    from score_trials import score, PLUGIN
    import importlib.util
    spec = importlib.util.spec_from_file_location("stats",PLUGIN)
    stats = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats)
    directory = HERE / "activation_search"
    directory.mkdir(exist_ok=True)
    cuda = Path(sys.prefix)/"lib/python3.12/site-packages/nvidia/cu13"
    headers = Path(sys.prefix)/"python-headers/usr/include"
    env = dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS="1",
             PATH=f"{Path(sys.prefix)/'bin'}:{cuda/'bin'}:{os.environ.get('PATH','')}",
             CPATH=f"{headers/'python3.12'}:{headers}:{os.environ.get('CPATH','')}",
             LD_LIBRARY_PATH=f"{cuda/'lib'}:/usr/lib/wsl/lib:{os.environ.get('LD_LIBRARY_PATH','')}")
    def run(group,seed):
        out=directory/f"g{group}_s{seed}.json"
        subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--group",str(group),"--seed",str(seed),
                        "--out",str(out)], env=env, check=True)
        r=score(out,stats)
        return dict(group=group,seed=seed,reward=r["reward"],gain=r.get("gain"),decision=r["decision"],file=out.name)
    # Explore all four choices on equal inputs. Repeat only the strongest resolved arms.
    trials = []
    for group in [8,32,4,16]:
        trials.append(run(group,59011))
        (directory/"trials.json").write_text(json.dumps(trials,indent=2)+"\n")
    eligible=sorted((x for x in trials if x["decision"]=="KEEP"),key=lambda x:x["reward"],reverse=True)
    for arm in eligible[:2]:
        trials.append(run(arm["group"],93481))
        (directory/"trials.json").write_text(json.dumps(trials,indent=2)+"\n")
    print(json.dumps(trials))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group",type=int,choices=[4,8,16,32])
    p.add_argument("--seed",type=int)
    p.add_argument("--out",type=Path)
    a=p.parse_args()
    if a.group:
        child(a.group,a.seed,a.out)
    else:
        search()
