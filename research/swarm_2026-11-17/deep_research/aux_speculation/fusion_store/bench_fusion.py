"""Paired graph timing of original kernel+copy vs added-store; local only."""
import argparse,importlib.util,json,random,statistics,sys,time
from pathlib import Path
import torch
from runtime_imports import prepare,S
prepare();P=Path(__file__).resolve().parent
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);a=pa.parse_args()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
base=load('timing_baseline_norm',S/'kernels/ops/layernorm/muse_spark_norm_residual_jit.py')
new=load('timing_aux_norm',P/'norm_residual_aux.py')
report={'scope':'Local RTX3050 graph primitive timing, not serving/H200 gain. Full existing outputs retained.','started':time.time(),'cases':[]}
def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
def same(x,y):return torch.equal(x.view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32),y.view(torch.int16 if y.dtype==torch.bfloat16 else torch.int32))
def timed(graph):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(40):graph.replay()
    end.record();end.synchronize();return start.elapsed_time(end)*1000/40
random.seed(41);torch.manual_seed(13)
for rows in [8,17,64,256,1024,4096]:
    h=8192;b=torch.randn(rows,h,device='cuda').to(torch.bfloat16);s=torch.randn(rows,h,device='cuda')
    weights=[torch.randn(h,device='cuda').to(torch.bfloat16) for _ in range(2)]
    kw=dict(post_eps=1e-6,pre_eps=1e-6,post_weight=weights[0],pre_weight=weights[1],alpha=torch.tensor([.75],device='cuda'),beta=torch.tensor([1.25],device='cuda'))
    outputs={};graphs={};buffers={}
    for variant in ['baseA','baseB','fused']:
        buffers[variant]=torch.empty(rows,h*5,dtype=torch.bfloat16,device='cuda')
        def run():
            if variant=='fused':return new.fused_norm_residual(b,s,**kw,aux_out=buffers[variant],aux_slot=2)
            ret=base.fused_norm_residual(b,s,**kw);buffers[variant][:,2*h:3*h].copy_(ret[1]);return ret
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):outputs[variant]=run()
        torch.cuda.current_stream().wait_stream(stream);graphs[variant]=graph
    for g in graphs.values():g.replay()
    assert all(same(outputs['baseA'][i],outputs['fused'][i]) for i in [0,1])
    assert same(buffers['baseA'][:,2*h:3*h],buffers['fused'][:,2*h:3*h])
    samples={name:[] for name in graphs}
    for _ in range(9):
        order=list(graphs);random.shuffle(order)
        for name in order:samples[name].append(timed(graphs[name]))
    med={name:statistics.median(vals) for name,vals in samples.items()}
    report['cases'].append({'rows':rows,'hidden':h,'exact_outputs':True,'samples_us':samples,'median_us':med,'base_over_fused':med['baseA']/med['fused'],'AA_ratio':med['baseA']/med['baseB']});save()
    del graphs,outputs,buffers,b,s,weights,kw
report.update(status='PASS',finished=time.time());save();print(json.dumps({'status':'PASS','cases':len(report['cases'])}))
