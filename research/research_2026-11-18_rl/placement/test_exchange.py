"""Validate real collective slot exchanges and pinned map publication with two Gloo workers."""
import ast
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from exchange import swap_slots,TENSORS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
MODEL=ROOT/"release_2026-11-18/candidate_sources/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py"


def hooks():
    tree=ast.parse(MODEL.read_bytes())
    cls=next(c for c in tree.body if getattr(c,"name","")=="MuseSpark1xFp4Experts")
    body=[n for n in cls.body if getattr(n,"name","") in ("refresh_expert_lookup_","update_expert_map_")]
    env={"torch":torch}
    exec(compile(ast.Module(body=body,type_ignores=[]),str(MODEL),"exec"),env)
    return env


def run(rank,init,folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",init_method=init,rank=rank,world_size=2)
    allslots=np.arange(8,dtype=np.int32).reshape(2,4)
    bank=torch.nn.Module()
    owner=torch.nn.Module()
    owner._humming_layer=bank
    env=hooks()
    for name in ("refresh_expert_lookup_","update_expert_map_"):
        setattr(owner,name,env[name].__get__(owner))
    owner.register_buffer("_expert_map",torch.full((8,),-1,dtype=torch.int32))
    owner.register_buffer("_expert_lookup",torch.empty(9,dtype=torch.int32))
    def material(idx,ids):
        cols=[32,16,1][idx%3]
        return (torch.from_numpy(ids.astype(np.int32)).reshape(-1,1)*31+
            torch.arange(cols)+idx*17).to([torch.int32,torch.uint8,torch.bfloat16][idx%3])
    for idx,name in enumerate(TENSORS):
        bank.register_buffer(name,material(idx,allslots[rank]))
    addresses=[t.data_ptr() for t in bank.buffers()]+[owner._expert_map.data_ptr(),owner._expert_lookup.data_ptr()]
    total=0
    for actions in [[(0,0,1,2),(0,1,1,3)],[(1,0,0,2)],[]]:
        allslots=swap_slots(owner,dist.group.WORLD,actions,before=allslots)
        for idx,name in enumerate(TENSORS):
            assert torch.equal(getattr(bank,name),material(idx,allslots[rank]))
            total+=1
        expect=torch.full((9,),-1,dtype=torch.int32)
        expect[torch.tensor(allslots[rank]+1,dtype=torch.long)]=torch.arange(4,dtype=torch.int32)
        assert torch.equal(expect,owner._expert_lookup)
        assert addresses==[t.data_ptr() for t in bank.buffers()]+[owner._expert_map.data_ptr(),owner._expert_lookup.data_ptr()]
    try:
        swap_slots(owner,dist.group.WORLD,[(0,rank,1,1)],before=allslots)
    except RuntimeError as e:
        assert "disagree" in str(e)
    else:raise AssertionError("Divergent peer plans must abort before weight writes")
    for idx,name in enumerate(TENSORS):
        assert torch.equal(getattr(bank,name),material(idx,allslots[rank]))
    dist.destroy_process_group()
    Path(folder,f"rank{rank}").write_text(json.dumps(dict(rank=rank,stage_checks=total,stable_addresses=True,divergence_rejected_before_mutation=True)))


if __name__=="__main__":
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(run,args=(Path(tmp,"init").as_uri(),tmp),nprocs=2,start_method="fork")
        rows=[json.loads(Path(tmp,f"rank{i}").read_text()) for i in range(2)]
        (HERE/"exchange_receipt.json").write_text(json.dumps({"backend":"two actual Gloo ranks, CPU; miniature shape. No NCCL/serving or timing claim.","results":rows},indent=2))
        print(rows)
