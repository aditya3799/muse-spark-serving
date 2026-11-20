"""Run a paired compact-width benchmark with the parent full-width expert tuning plan."""
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import sys

here=Path(__file__).resolve().parent
scope=here.parents[1]/"research_2026-11-18_rl/lattice_tree"
sys.path.insert(0,str(scope))
import width_gpu as original
from sglang.kernels.ops.moe import muse_spark_humming_split as split

p=argparse.ArgumentParser(add_help=False)
p.add_argument("--method",choices=("pin_config","dense_histogram"),required=True)
p.add_argument("--stress-crossing",action="store_true")
custom,others=p.parse_known_args();sys.argv=[sys.argv[0],*others]
real_choose=split.choose_partition_configs
real_partition=split.partition_routes
def repaired(fn,layer,b,x,ids,s,dense_ids):
    # Keep valid_shape_m and allocations at actual compact size; only choose the
    # config table interval using the rows of the reference width-eight mode.
    def choose(layer,actual_routes,global_experts):
        assert actual_routes==ids.numel()
        return real_choose(layer,b*8*8,global_experts)
    def partition(local_ids,experts,cutoff):
        if custom.method=="pin_config":return real_partition(local_ids,experts,cutoff)
        # Diagnostic oracle; uncomputed rows' true router results are not available
        # in a compact serving pass. Fixed-shape device operations support capture.
        _,_,counts=real_partition(dense_ids,experts,cutoff)
        labels=counts[local_ids.long().clamp_min(0)]
        light=(local_ids>=0)&(labels<=cutoff)
        heavy=(local_ids>=0)&(labels>cutoff)
        return original.torch.where(light,local_ids,-1),original.torch.where(heavy,local_ids,-1),counts
    with patch.object(split,"choose_partition_configs",choose),patch.object(split,"partition_routes",partition):
        return fn(x,ids,s)
def install():
    source=(scope/"width_gpu.py").read_text()
    source=source[source.index("def evaluate("):source.index("\ndef main(")]
    old="return run(input_x.index_select(0,rows),ids,input_w.index_select(0,rows))"
    new="return repaired(run,layer,b,input_x.index_select(0,rows),ids,input_w.index_select(0,rows),input_ids)"
    assert source.count(old)==1
    source=source.replace(old,new)
    if custom.stress_crossing:
        # Give global expert 0 one live slot on every final request-position
        # plus exactly one first row. Original B16/B32 count is B+1; for widths
        # below eight most last positions disappear; all other slots remote.
        anchor="input_ids.copy_(torch.where(ids<32,ids,-1));"
        assert source.count(anchor)==1
        source=source.replace(anchor,"input_ids.fill_(-1);input_ids[7::8,0]=0;input_ids[0,0]=0;")
    ns=dict(original.__dict__,repaired=repaired)
    exec(compile(source,str(here/"repair_widths.py"),"exec"),ns)
    original.evaluate=ns["evaluate"]
if __name__=="__main__":
    install()
    with original.torch.inference_mode():
        original.main()
