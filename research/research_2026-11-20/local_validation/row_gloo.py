"""Real Gloo collective-order and distinct-KV-head contract test; no CUDA math claim."""
import argparse,json,os,tempfile,uuid
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys,types,importlib
SOURCE=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_final/assembled/python/sglang')
for name,path in [('sglang',SOURCE),('sglang.srt',SOURCE/'srt'),('sglang.srt.speculative',SOURCE/'srt/speculative')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
mod=importlib.import_module('sglang.srt.speculative.dflash_projection_row')
RowProjectionRouter,discover=mod.RowProjectionRouter,mod.discover
HERE=Path(__file__).resolve().parent

def worker(rank,world,path,out):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,world_size=world,rank=rank)
    class Group:
        world_size=world;rank_in_group=rank;cpu_group=dist.group.WORLD
        def __init__(self):self.calls=0
        def all_gather(self,x,dim):
            self.calls+=1;parts=[torch.empty_like(x) for _ in range(world)];dist.all_gather(parts,x)
            return torch.cat(parts,dim=dim)
    gen=torch.Generator().manual_seed(9137)
    linear=torch.nn.Linear(16,16,bias=False,dtype=torch.bfloat16)
    with torch.no_grad():linear.weight.copy_(torch.randint(-2,3,linear.weight.shape,generator=gen)*.125)
    x=(torch.randint(-2,3,(512,16),generator=gen)*.125).bfloat16();reference=linear(x)
    class Plan:
        def __init__(self,reject=False,wrong=False):self.reject,self.wrong=reject,wrong
        def local(self,value,weight,rank_):
            if self.reject:return None
            output=torch.nn.functional.linear(value,weight).chunk(world,dim=0)[rank_].clone()
            if self.wrong:output[0,0]+=1
            return output
        def close(self):pass
    receipts={}
    for scenario in ('ordinary','one_rank_declines_registration','one_rank_discovery_exception','one_rank_local_fallback','strict_bad_rank'):
        group=Group();router=RowProjectionRouter(linear,group,empirical=scenario!='strict_bad_rank')
        accepted,_=router.prepare_registration([x.shape]);assert accepted
        def discover(examples,size):
            if scenario=='one_rank_declines_registration' and rank==world-1:return None,{}
            if scenario=='one_rank_discovery_exception' and rank==world-1:raise RuntimeError('injected local discovery failure')
            return Plan(reject=scenario=='one_rank_local_fallback' and rank==world-1,
                        wrong=scenario=='strict_bad_rank' and rank==0),{}
        try:router.register([x],discover_fn=discover)
        except RuntimeError as error:
            assert scenario=='one_rank_discovery_exception' and 'qualification aborted' in str(error)
            assert group.calls==0
            receipts[scenario]=dict(coherent_fatal_abort=True,gather_calls=0)
            router.close();continue
        assert scenario!='one_rank_discovery_exception', 'fatal error silently swallowed'
        result=router(x);assert torch.equal(result.view(torch.int16),reference.view(torch.int16))
        assert group.calls==(0 if scenario in ('one_rank_declines_registration','one_rank_discovery_exception') else 1)
        for m in (1,4,16):
            before=group.calls;assert torch.equal(router(x[:m]),linear(x[:m]));assert group.calls==before
        receipts[scenario]=dict(exact=True,gather_calls=group.calls,decode_rows_1_4_16_unchanged=True)
        router.close()
    group=Group();router=RowProjectionRouter(linear,group,empirical=True)
    accepted,_=router.prepare_registration([x.shape] if rank==0 else [(768,16)])
    assert not accepted
    metadata=router.register([x] if rank==0 else [])
    assert not metadata['accepted'] and not router.published
    assert torch.equal(router(x),reference) and group.calls==0
    receipts['mismatched_manifest_and_empty_rank']=dict(coherent_fallback=True)
    router.close()
    # Each rank owns a different KV head. FC-gather followed by the original
    # own-rank projection is exact; gathering already projected rows mixes heads.
    own_weight=torch.full((4,16),(rank+1)*.125,dtype=torch.bfloat16)
    own_full=torch.nn.functional.linear(reference,own_weight)
    own_local=torch.nn.functional.linear(reference.chunk(world,0)[rank],own_weight)
    wrong=Group().all_gather(own_local,dim=0)
    assert not torch.equal(wrong,own_full)
    receipts['distinct_head_counterexample']=dict(invalid_packed_kv_gather_mismatches=int(torch.count_nonzero(wrong!=own_full)))
    if rank==0:Path(out).write_text(json.dumps(dict(world=world,cases=receipts,correct=True,scope='Real Gloo CPU transport with exact integer fixture; vendor plans mocked, CUDA math not exercised'),indent=2)+'\n')
    dist.destroy_process_group()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worlds',type=int,nargs='+',default=[2,4]);args=parser.parse_args()
    previous=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        plan,reason=discover([],8)
        assert plan is None and 'preference mask' in reason['reason']
        (HERE/'row_policy_gate.json').write_text(json.dumps(dict(strict_bf16_policy_rejected=True,reason=reason),indent=2)+'\n')
    finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=previous
    for world in args.worlds:
        path=str(Path(tempfile.gettempdir())/('okc-row-'+uuid.uuid4().hex))
        mp.start_processes(worker,args=(world,path,str(HERE/f'row_adapter_cpu_tp{world}.json')),nprocs=world,join=True,start_method='fork')
        print('Gloo',world,'passed',flush=True)
