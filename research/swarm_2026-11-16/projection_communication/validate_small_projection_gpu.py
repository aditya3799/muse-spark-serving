"""Exact small-column arithmetic plus cold-weight component timings."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,statistics
import torch
import torch.nn.functional as F
P=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/projection_communication')
spec=importlib.util.spec_from_file_location('validation',P/'validate_row_projection.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
H=P/'candidate_small_multicast/python/sglang/srt/layers/communication/muse_spark_gemm_ag.py'
local=v.extract(H,{'small_projection_local'},{'torch':torch})['small_projection_local']
parser=argparse.ArgumentParser();parser.add_argument('--numerical-only',action='store_true');parser.add_argument('--timing-only',action='store_true');args=parser.parse_args()
prior=json.loads((P/'small_gpu_numerical.json').read_text()) if args.timing_only else None
if prior: assert prior['helper_sha256']==hashlib.sha256(H.read_bytes()).hexdigest()
rows=[]
with torch.inference_mode():
 evict=None if args.numerical_only else torch.zeros(128*1024*1024//4,device='cuda',dtype=torch.float32)
 for k,n in ((8192,4096),(4096,8192)):
    torch.manual_seed(711)
    weights=[(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16) for _ in range(1 if args.numerical_only else 8)]
    for m in (1,8,16,32,64,128,256,511):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
        if args.timing_only:
            errors=next(r['errors'] for r in prior['rows'] if (r['m'],r['k'],r['n'])==(m,k,n))
        else:
            w=weights[0]
            noncontig=m==32
            backing=torch.randn(m,k*(2 if noncontig else 1),device='cuda',dtype=torch.bfloat16)
            x=backing[:,::2] if noncontig else backing
            gg,combined=v.graph(lambda:torch.cat([local(SimpleNamespace(world_size=8,rank=r),x,w) for r in range(8)],dim=1))
            errors=[]
            for reduction in (False,True):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduction
                for seed in (23,41,67):
                    torch.manual_seed(seed);x.copy_(torch.randn_like(x));gg.replay()
                    baseline=F.linear(x,w);ref=F.linear(x.float(),w.float())
                    rel=((combined.float()-baseline.float()).norm()/baseline.float().norm()).item()
                    maxabs=(combined.float()-baseline.float()).abs().max().item()
                    cand_ref=((combined.float()-ref).norm()/ref.norm()).item()
                    base_ref=((baseline.float()-ref).norm()/ref.norm()).item()
                    errors.append(dict(seed=seed,reduction=reduction,noncontiguous=noncontig,relative_l2=rel,max_abs=maxabs,candidate_ref_l2=cand_ref,baseline_ref_l2=base_ref,passed=rel<=0.003 and maxabs<=0.03125 and cand_ref<=base_ref+1e-5))
                    del baseline,ref
            del gg,combined,x,backing
        if args.numerical_only:
            rows.append(dict(m=m,k=k,n=n,errors=errors))
            print(json.dumps(dict(m=m,k=k,n=n,passed=all(e['passed'] for e in errors))),flush=True)
            continue
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
        gathered=[torch.empty(8,m,n//8,device='cuda',dtype=torch.bfloat16) for _ in weights]
        comm=SimpleNamespace(world_size=8,rank=0)
        calls={'full':lambda:[F.linear(x,wi) for wi in weights], 'full_control':lambda:[F.linear(x,wi) for wi in weights], 'local_fp32':lambda:[local(comm,x,wi) for wi in weights], 'column_layout':lambda:[g.movedim(0,1).reshape(m,n) for g in gathered]}
        graphs={};values=[]
        for name,call in calls.items():g,y=v.graph(call);graphs[name]=g;values.append(y)
        samples=[]
        for i in range(8):
            order=list(graphs)
            if i%2:order.reverse()
            sample={}
            for name in order:
                evict.add_(1)
                sample[name]=v.elapsed(graphs[name],reps=1)/len(weights)
            samples.append(sample)
        med={name:statistics.median(s[name] for s in samples) for name in graphs}
        row=dict(m=m,k=k,n=n,serving_branch_changed=not(k==4096 and n==8192 and m<=8),errors=errors,median_us=med,samples=samples,compute_only_communication_budget_us=med['full']-med['local_fp32'],nccl_layout_budget_us=med['full']-med['local_fp32']-med['column_layout'])
        rows.append(row)
        print(json.dumps(dict(m=m,k=k,n=n,passed=all(e['passed'] for e in errors),median_us=med)),flush=True)
        del graphs,values,g,y,x,gathered,calls
        torch.cuda.empty_cache()
    del weights
    torch.cuda.empty_cache()
report=dict(rows=rows,passed=all(e['passed'] for r in rows for e in r['errors']),helper_sha256=hashlib.sha256(H.read_bytes()).hexdigest(),numerical_only=args.numerical_only,weight_rotation=1 if args.numerical_only else 8,l2_eviction_bytes=0 if args.numerical_only else 128*1024*1024,full_weight_rotation_bytes=(1 if args.numerical_only else 8)*8192*4096*2,local_weight_rotation_bytes=(1 if args.numerical_only else 8)*8192*4096*2//8,gpu=torch.cuda.get_device_name(),scope='128MiB read/write eviction before each timing plus rotating-weight RTX3050 compute components; exact local FP32 helper, serial rank assembly for numerics. No real multicast/NCCL, no H200 or serving speedup measurement. Native multicast includes layout without separate transpose; column_layout is NCCL alternative cost only.')
(P/('small_gpu_numerical.json' if args.numerical_only else 'small_gpu_validation_and_timings.json')).write_text(json.dumps(report,indent=2))
assert report['passed'],'small projection numerical screen failed'
