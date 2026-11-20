from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, json, sys
import torch
import torch.nn.functional as F
test_path=Path('/mnt/d/OpenKernelChallenge/serving/research_iteration_2026-11-15/projection_candidate/validate_projection_candidate.py')
spec=importlib.util.spec_from_file_location('projection_validation',test_path)
validation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)
load,graph,OUT,HELPER=validation.load,validation.graph,validation.OUT,validation.HELPER

helper,_=load()
class SerialRanks:
    world_size=8
    rank_in_group=3
    def __init__(self,x,w): self.x,self.w=x,w
    def all_gather(self,local,dim):
        # A graph-capturable mathematical stand-in for communication only.
        # CPU tests separately exercised the real distributed gather/layout.
        parts=[local if rank==self.rank_in_group else F.linear(self.x,wi)
               for rank,wi in enumerate(self.w.chunk(self.world_size))]
        return torch.cat(parts,dim=dim)

rows=[]
with torch.inference_mode():
    for reduced in (False, True):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
        for k,n in ((8192,4096),(4096,8192)):
            for m in (512,777):
                torch.manual_seed(41+m)
                w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
                storage=torch.randn(m,k*2,device='cuda',dtype=torch.bfloat16)
                x=storage[:,::2]
                group=SerialRanks(x,w)
                g,out=graph(lambda:helper.prefill_projection(group,x,w))
                max_error=0.;max_l2=0.
                for seed in (73,101,307):
                    torch.manual_seed(seed)
                    storage.normal_()
                    g.replay()
                    reference=F.linear(x,w)
                    diff=(out.float()-reference.float())
                    rel=(diff.norm()/reference.float().norm()).item()
                    error=diff.abs().max().item()
                    max_error=max(max_error,error);max_l2=max(max_l2,rel)
                rows.append(dict(m=m,k=k,n=n,reduced_precision_reduction=reduced,
                    noncontiguous_input=True,replays=3,max_abs=max_error,max_relative_l2=max_l2,
                    passed=max_l2<0.003 and max_error<=0.03125))
                del g,out,group,storage,x,w,reference,diff
report=dict(passed=all(r['passed'] for r in rows),rows=rows,
    screening_thresholds=dict(relative_l2_lt=0.003,max_abs_le=0.03125),
    helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
    scope='exact exported projection helper with serial-rank GEMMs in place of network; changed-input CUDA graph math/layout check, NOT an NCCL test')
(OUT/'gpu_replay_checks.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
