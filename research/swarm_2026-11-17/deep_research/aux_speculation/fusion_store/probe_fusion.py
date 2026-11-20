"""Exact baseline-versus-store primitive probe; requires root hardware lease."""
import argparse,contextlib,importlib.util,json,time,sys
from pathlib import Path
import torch
from runtime_imports import prepare,S
prepare()
P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
base=load('baseline_norm_residual',S/'kernels/ops/layernorm/muse_spark_norm_residual_jit.py')
new=load('candidate_norm_residual',P/'norm_residual_aux.py')
checks=[]
report={'scope':'Actual pinned JIT norm/residual kernel against isolated added-store kernel; no full model integration, no H200 runtime.', 'started':time.time(),'device':torch.cuda.get_device_name(),'checks':checks,'timings':[]}
def save(): a.out.write_text(json.dumps(report,indent=2)+'\n')
def check(name,passed,**kw):
    checks.append(dict(name=name,passed=bool(passed),**kw));save()
    if not passed: raise AssertionError(name)
def bits(x): return x.view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32)
def same(x,y):return torch.equal(bits(x),bits(y))
torch.manual_seed(1794)
hidden=8192
for rows in [17,256]:
    branch=torch.randn(rows,hidden,device='cuda').to(torch.bfloat16)
    stream=torch.randn(rows,hidden,device='cuda')*3
    post=torch.randn(hidden,device='cuda').to(torch.bfloat16)
    pre=torch.randn(hidden,device='cuda').to(torch.bfloat16)
    alpha=torch.randn(hidden,device='cuda')*.1+1
    beta=torch.randn(hidden,device='cuda')*.1+1
    for case in range(8):
        kw=dict(post_eps=1e-6,pre_eps=1e-6,post_norm_is_input_scale=bool(case&1),
            post_weight=post if case&2 else None,pre_weight=pre if case&2 else None,
            alpha=alpha if case&4 else None,beta=beta if case&4 else None,
            want_router=bool(case&1),pre_norm_bf16=bool(case&2))
        packed=torch.full((rows,hidden*5),-77,dtype=torch.bfloat16,device='cuda');slot=[0,1,4][case%3]
        before=time.time();ref=base.fused_norm_residual(branch,stream,**kw);got=new.fused_norm_residual(branch,stream,**kw,aux_out=packed,aux_slot=slot)
        torch.cuda.synchronize();report['timings'].append({'rows':rows,'case':case,'wall_s':time.time()-before})
        for i,(x,y) in enumerate(zip(ref,got)):
            check('m%d_case%d_output%d'%(rows,case,i),x is None and y is None or x is not None and y is not None and same(x,y))
        check('m%d_case%d_aux'%(rows,case),same(packed[:,slot*hidden:(slot+1)*hidden],ref[1].to(torch.bfloat16)))
        other=[i for i in range(5) if i!=slot]
        check('m%d_case%d_neighbor_sentinels'%(rows,case),all(bool((packed[:,i*hidden:(i+1)*hidden]==-77).all()) for i in other))
    # Capture the true kernel and update inputs without recompiling.
    kw=dict(post_eps=1e-6,pre_eps=1e-6,post_weight=post,pre_weight=pre,alpha=alpha,beta=beta)
    packed=torch.empty(rows,hidden*5,dtype=torch.bfloat16,device='cuda')
    stream_capture=torch.cuda.Stream();stream_capture.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream_capture):
        new.fused_norm_residual(branch,stream,**kw,aux_out=packed,aux_slot=4)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream_capture):got=new.fused_norm_residual(branch,stream,**kw,aux_out=packed,aux_slot=4)
    torch.cuda.current_stream().wait_stream(stream_capture)
    for cycle in range(3):
        branch.copy_(torch.randn_like(branch));stream.copy_(torch.randn_like(stream)*(cycle+1))
        graph.replay();ref=base.fused_norm_residual(branch,stream,**kw)
        check('graph_m%d_cycle%d_outputs'%(rows,cycle),same(ref[0],got[0]) and same(ref[1],got[1]))
        check('graph_m%d_cycle%d_aux'%(rows,cycle),same(packed[:,4*hidden:],ref[1].to(torch.bfloat16)))
# Scalar gates and conversion edge values use already-compiled small-M policy.
rows=17
branch=torch.zeros(rows,hidden,dtype=torch.bfloat16,device='cuda')
edge=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),1.00390625,-1.00390625,1e-40,-1e-40],dtype=torch.float32,device='cuda')
stream=edge.repeat((rows*hidden+8)//9)[:rows*hidden].reshape(rows,hidden)
for scalar_gate in [False,True]:
    kw=dict(post_eps=1e-6,pre_eps=1e-6,alpha=torch.tensor([.75],device='cuda') if scalar_gate else None,beta=torch.tensor([1.25],device='cuda') if scalar_gate else None)
    packed=torch.full((rows,hidden*5),-77,dtype=torch.bfloat16,device='cuda')
    ref=base.fused_norm_residual(branch,stream,**kw);got=new.fused_norm_residual(branch,stream,**kw,aux_out=packed,aux_slot=3)
    check('conversion_edge_outputs_scalar'+str(scalar_gate),same(ref[0],got[0]) and same(ref[1],got[1]))
    check('conversion_edge_aux_scalar'+str(scalar_gate),same(packed[:,3*hidden:4*hidden],ref[1].to(torch.bfloat16)))
report.update(status='PASS',finished=time.time(),check_count=len(checks));save();print(json.dumps({'status':'PASS','checks':len(checks)}))
