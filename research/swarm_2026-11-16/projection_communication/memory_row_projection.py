"""Allocation/capture audit of repeated projections; collective payload is mocked."""
from pathlib import Path
import contextlib,importlib.util,json,logging
import torch
import torch.nn.functional as F
P=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication')
spec=importlib.util.spec_from_file_location('validation',P/'validate_row_projection.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
run=v.helper()
ns=dict(torch=torch,logger=logging.getLogger('memory'))
gather=v.extract(v.ROOT/'sglang/python/sglang/srt/distributed/parallel_state.py',{'all_gather'},ns,'GroupCoordinator')['all_gather']
class Group:
    world_size=8
    rank_in_group=0
    hpu_communicator=None
    def use_symmetric_memory(self,*a,**kw):return contextlib.nullcontext()
    def is_allocation_symmetric(self):return False
    def all_gather_into_tensor(self,output,input):
        # No extra materialization: simulate destination writes into exact gather buffer.
        for chunk in output.chunk(self.world_size,dim=0):chunk.copy_(input)
    def all_gather(self,x,dim):return gather(self,x,dim=dim)
group=Group()
rows=[]
with torch.inference_mode():
    torch.manual_seed(51)
    for m in (512,16384):
        x=torch.randn(m,8192,device='cuda',dtype=torch.bfloat16)
        pool=(torch.randn(4096,8192,device='cuda')/8192**.5).to(torch.bfloat16)
        unpool=(torch.randn(8192,4096,device='cuda')/4096**.5).to(torch.bfloat16)
        for mode in ('full','row'):
            for layers in (1,64):
                torch.cuda.synchronize();torch.cuda.empty_cache()
                base_alloc=torch.cuda.memory_allocated();base_reserved=torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                def chain():
                    y=x
                    for _ in range(layers):
                        if mode=='full':y=F.linear(F.linear(y,pool),unpool)
                        else:y=run(group,run(group,y,pool),unpool)
                    return y
                graph,out=v.graph(chain)
                graph.replay();graph.replay();torch.cuda.synchronize()
                row=dict(m=m,mode=mode,layers=layers,base_allocated=base_alloc,base_reserved=base_reserved,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),current_allocated=torch.cuda.memory_allocated(),current_reserved=torch.cuda.memory_reserved())
                row['peak_extra_allocated']=row['peak_allocated']-base_alloc
                rows.append(row)
                print(json.dumps(row),flush=True)
                del graph,out
                torch.cuda.synchronize();torch.cuda.empty_cache()
        del x,pool,unpool
        torch.cuda.empty_cache()
report=dict(rows=rows,gpu=torch.cuda.get_device_name(),scope='Shared-weight 1/64-layer projection chain; exact helper/source-extracted GroupCoordinator gather layout and allocation; simulated gather buffer writes. Excludes NCCL/multiGPU workspace, full model, simultaneous graph sizes and real serving peak.')
(P/'gpu_memory_chain.json').write_text(json.dumps(report,indent=2))
