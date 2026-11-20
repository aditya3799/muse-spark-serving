"""Full-width vs eight serial shards and one diagnostic shard; no distributed transport."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from runtime_projection import discover, native_workspace
from audit_match import BASELINE, PARENT, SHARD, N, K, mismatches
import hashlib

def run(args):
    digest=hashlib.sha256(PARENT.read_bytes()).hexdigest()
    assert digest==BASELINE
    generator=torch.Generator(device="cuda").manual_seed(args.seed)
    host_random=random.Random(args.seed)
    props=torch.cuda.get_device_properties(0)
    record=dict(seed=args.seed, patch_digest=digest, hardware=str(props), runtime_ws=native_workspace(),
                scope="one GPU, eight serial virtual ranks and torch.cat; one-shard timing omits all transport",
                shapes={})
    def tensor(*shape,std=1):
        return torch.empty(*shape,dtype=torch.bfloat16,device="cuda").normal_(std=std,generator=generator)
    weights=tensor(N,K,std=.006),tensor(N,K,std=.006)
    methods=["baseline","control","serial8_cat","one_shard"]
    for m in args.rows:
        examples=[(tensor(m,K),w) for w in weights]
        torch.cuda.synchronize()
        start=time.perf_counter()
        plan,info=discover(examples,8)
        torch.cuda.synchronize()
        info["startup_ms"]=1000*(time.perf_counter()-start)
        record["shapes"][m]=info
        if plan is not None:
            # Unseen weights and inputs after plan selection.
            fresh=tensor(N,K,std=.006)
            feeds=[tensor(m,K) for _ in range(3)]
            def serial(v): return torch.cat([plan.local(v,fresh,r) for r in range(8)],dim=1)
            def baseline(v): return F.linear(v,fresh)
            functions={"baseline":baseline,"control":baseline,"serial8_cat":serial,
                       "one_shard":lambda v:plan.local(v,fresh,0)}
            info["holdout_mismatch_counts"]=[mismatches(serial(v),baseline(v)) for v in feeds]
            info["timings_us"]={name:[] for name in methods}
            info["orders"]=[]
            for v in feeds:
                for fn in functions.values(): fn(v)
            for pair in range(args.pairs):
                order=methods.copy()
                host_random.shuffle(order)
                v=feeds[pair%len(feeds)]
                for direction in (order,list(reversed(order))):
                    info["orders"].append(direction)
                    for label in direction:
                        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        a.record()
                        for _ in range(args.inner): functions[label](v)
                        b.record()
                        b.synchronize()
                        info["timings_us"][label].append(a.elapsed_time(b)*1000/args.inner)
            info["median_us"]={n:statistics.median(v) for n,v in info["timings_us"].items()}
            plan.close()
            del fresh,feeds
        args.out.write_text(json.dumps(record,indent=2)+"\n")
        print(m,info.get("chosen",info.get("reason")),info["startup_ms"], info.get("median_us"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows",type=int,nargs="+",default=[8,32,128,256,753])
    p.add_argument("--seed",type=int,default=419857)
    p.add_argument("--pairs",type=int,default=5)
    p.add_argument("--inner",type=int,default=3)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())

