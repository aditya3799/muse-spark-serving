from pathlib import Path
import hashlib, importlib.util, json, statistics
import torch
import torch.nn.functional as F

ROOT=Path('/mnt/d/OpenKernelChallenge/serving/research_iteration_2026-11-15')
OUT=ROOT/'projection_fp32_candidate'
HELPER=OUT/'candidate_sources/python/sglang/srt/layers/communication/muse_spark_prefill_projection.py'
spec=importlib.util.spec_from_file_location('strict_projection',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
spec=importlib.util.spec_from_file_location('validation',ROOT/'projection_candidate/validate_projection_candidate.py')
validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)

class SerialRanks:
    world_size=8
    rank_in_group=3
    def __init__(self,x,w): self.x,self.w=x,w
    def all_gather(self,local,dim):
        return torch.cat([local if r==3 else torch.mm(self.x,wi.t(),out_dtype=torch.float32).to(self.x.dtype)
            for r,wi in enumerate(self.w.chunk(8))],dim=dim)

rows=[]
with torch.inference_mode():
    torch.backends.cuda.matmul.allow_tf32=False
    for reduced in (False,True):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
        for k,n in ((8192,4096),(4096,8192)):
            torch.manual_seed(41)
            w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
            for m in (512,513,777,4096,16384):
                storage=torch.empty(m,k*2,device='cuda',dtype=torch.bfloat16)
                x=storage[:,::2];storage.normal_()
                group=SerialRanks(x,w)
                g,actual=validation.graph(lambda:helper.prefill_projection(group,x,w))
                samples=[]
                for seed in (73,101,307):
                    torch.manual_seed(seed);storage.normal_();g.replay()
                    baseline=F.linear(x,w)
                    oracle=F.linear(x.float(),w.float())
                    d=actual.float()-baseline.float()
                    rel=(d.norm()/baseline.float().norm()).item()
                    maxabs=d.abs().max().item()
                    candidate_l2=((actual.float()-oracle).norm()/oracle.norm()).item()
                    baseline_l2=((baseline.float()-oracle).norm()/oracle.norm()).item()
                    passed=(rel<0.003 and maxabs<=0.03125 and candidate_l2<=baseline_l2+0.00001)
                    samples.append(dict(seed=seed,relative_l2=rel,maxabs=maxabs,
                        candidate_vs_fp32_l2=candidate_l2,baseline_vs_fp32_l2=baseline_l2,passed=passed))
                row=dict(m=m,k=k,n=n,reduced=reduced,noncontiguous=True,checks=samples)
                rows.append(row)
                (OUT/'gpu_validation.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2))
                print(json.dumps(dict(m=m,k=k,n=n,reduced=reduced,passed=all(s['passed'] for s in samples))),flush=True)
                del g,actual,group,storage,x,baseline,oracle,d
            del w
report=dict(status='passed' if all(s['passed'] for r in rows for s in r['checks']) else 'failed',rows=rows,
    gpu=torch.cuda.get_device_name(),helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
    scope='Exact exported helper; CUDA graphs with changed/noncontiguous inputs; serial rank GEMMs stand in for network. Not an NCCL test.',
    thresholds=dict(relative_l2_lt=0.003,maxabs_le=0.03125,extra_error_vs_fp32_le=0.00001))
(OUT/'gpu_validation.json').write_text(json.dumps(report,indent=2))
print(report['status'])
