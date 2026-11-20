from pathlib import Path
import importlib.util,json,statistics,torch
import torch.nn.functional as F
root=Path('/mnt/d/OpenKernelChallenge/serving/research_iteration_2026-11-15')
out=root/'projection_fp32_candidate'
spec=importlib.util.spec_from_file_location('old',root/'projection_candidate/validate_projection_candidate.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
rows=[]
with torch.inference_mode():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    for k,n in ((8192,4096),(4096,8192)):
        w=(torch.randn(n,k,device='cuda')/k**.5).to(torch.bfloat16)
        for m in (512,1024,4096,16384):
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            gathered=torch.empty(8,m,n//8,device='cuda',dtype=torch.bfloat16)
            graphs={}
            values=[]
            for key,call in [
                ('full',lambda:F.linear(x,w)),
                ('full_control',lambda:F.linear(x,w)),
                ('local_fp32',lambda:torch.mm(x,w[:n//8].t(),out_dtype=torch.float32).to(x.dtype)),
                ('layout',lambda:gathered.movedim(0,1).reshape(m,n)),
            ]:
                g,value=mod.graph(call);graphs[key]=g;values.append(value)
            samples=[]
            for i in range(8):
                names=list(graphs)
                if i%2:names.reverse()
                samples.append({name:mod.elapsed(graphs[name],3 if m>=4096 else 10) for name in names})
            med={name:statistics.median(s[name] for s in samples) for name in graphs}
            row=dict(m=m,k=k,n=n,median_us=med,samples=samples,
                communication_budget_us=med['full']-med['local_fp32']-med['layout'],
                paired_control_ratio=statistics.median(s['full']/s['full_control'] for s in samples))
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='samples'}),flush=True)
            del graphs,values,x,gathered,g,value
report=dict(rows=rows,gpu=torch.cuda.get_device_name(),
    scope='RTX3050 compute and layout costs only; NCCL absent. Communication budget is local, not an H200 estimate; no serving speedup measured.')
(out/'gpu_component_timings.json').write_text(json.dumps(report,indent=2))
