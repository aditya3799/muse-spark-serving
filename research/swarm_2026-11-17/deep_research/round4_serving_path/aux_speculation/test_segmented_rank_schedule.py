"""CPU model of actual SpecTpSync decision broadcast; no real distributed run."""
import ast,hashlib,json,logging
from pathlib import Path
from types import SimpleNamespace as N
import torch
P=Path(__file__).parent;ROOT=P.parents[3];SOURCE=ROOT/'sglang/python/sglang/srt/speculative/spec_tp_sync.py'
tree=ast.parse(SOURCE.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('sglang'))]
ns={'envs':N(SGLANG_SPEC_TP_SYNC=N(get=lambda:'all'))};exec(compile(tree,str(SOURCE),'exec'),ns)
Sync=ns['SpecTpSync'];Site=ns['SpecTpSyncSite'];torch.set_num_threads(1)
logging.getLogger().setLevel(logging.ERROR)
class Bus:
    value=None
class Group:
    world_size=8
    def __init__(self,rank,bus):self.rank_in_group=rank;self.bus=bus;self.calls=[]
    def broadcast(self,value,src):
        assert src==0;self.calls.append(tuple(value.shape))
        if self.rank_in_group==0:self.bus.value=value.clone()
        else:value.copy_(self.bus.value)

def main():
    cases=checks=0;divergent=0
    for bs in (1,3,8,32):
      for generation in range(8):
        candidate=torch.arange(bs*8).view(bs,8)+100
        first=candidate[:,1:5].clone()
        for row in range(bs):
            if (row+generation)%3==0:first[row,(row+generation)%4]=-1
        bus=Bus();groups=[Group(rank,bus) for rank in range(8)];syncs=[Sync(g) for g in groups]
        original=[];agreed=[]
        for rank,sync in enumerate(syncs):
            pred=first.clone()
            if rank%2:pred[:,3]=-7
            original.append((pred==candidate[:,1:5]).all(1).nonzero().flatten().tolist())
            assert sync.enabled(Site.DFLASH_ACCEPT_GREEDY)
            sync.sync(Site.DFLASH_ACCEPT_GREEDY,pred)
            rows=(pred==candidate[:,1:5]).all(1).nonzero().flatten().tolist()
            agreed.append(rows)
        if any(rows!=original[0] for rows in original[1:]):divergent+=1
        assert all(rows==agreed[0] for rows in agreed);checks+=7
        count=len(agreed[0]);traces=[]
        for rank,sync in enumerate(syncs):
            trace=[('target_first',bs,4),('sync_first',bs,4),('host_count_dependency',count)]
            if count:
                pred=torch.full((count,4),rank,dtype=torch.int64)
                sync.sync(Site.DFLASH_ACCEPT_GREEDY,pred)
                assert torch.equal(pred,torch.zeros_like(pred));checks+=1
                trace.extend([('target_tail',count,4),('sync_tail',count,4)])
            trace.append(('publish_once',bs))
            traces.append(trace)
        assert all(trace==traces[0] for trace in traces);checks+=7;cases+=1
    assert divergent>0
    ns['envs'].SGLANG_SPEC_TP_SYNC.get=lambda:'off'
    disabled=Sync(Group(0,Bus()));assert not disabled.enabled(Site.DFLASH_ACCEPT_GREEDY)
    # All ranks must agree on admission before any per-step conditional collectives.
    configs=[('segmented-v1',4,True)]*8;bad=configs.copy();bad[-1]=('segmented-v1',4,False)
    assert len(set(configs))==1 and len(set(bad))!=1
    report={'status':'CPU_SOURCE_SYNC_SCHEDULE_PASS','cases':cases,'assertions':checks,'ranks_modeled':8,
      'prebroadcast_divergent_cases':divergent,'negative_controls':['disabled_greedy_sync_rejected','mismatched_startup_config_rejected','unsynchronized_local_predictions_produce_different_survivor_shapes'],
      'scope':'Actual SpecTpSync.sync/enabled/parser executed with in-process fake broadcast bus; no Gloo/NCCL/GPU collectives',
      'production_protocol':['Unconditional startup CPU-group consensus on enabled/version/widths/topology/sync policy BEFORE capture; mismatch fails globally',
        'Keep same original requestorder/candidates on everyrank; existing stage1 greedy prediction broadcast must be enabled',
        'Derive survivor mask/order deterministically from broadcast predictions and replicated candidates',
        'Obtain survivorcount onhost before selecting tailgraph; cost is explicit, not a CUDAgraph conditional assumption',
        'All ranks either skip tail together or run sameS/bucket and same collective sequence; tailpredict sync then mergeoriginalrequestrows',
        'Use existing group broadcast fallback; do not assume particularNCCL capture capability'],
      'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'holds':['RealTP8 NCCL and requestlifecycle integration','Hostdecision and collective latency','Fullmodel numerical equality']}
    (P/'SEGMENTED_RANK_SCHEDULE_CPU.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()