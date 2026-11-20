"""Execute exact exported wrapper with source-path relocation only."""
import argparse,ast,hashlib,importlib.util,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sglang-root',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
from kernel_runtime import bootstrap
bootstrap(a.sglang_root)
import torch
from sglang.kernels.ops.activation.activation import silu_and_mul
EXPORT=P/'scale_activation_export/candidate_sources'
wrapper=EXPORT/'python/sglang/kernels/ops/moe/muse_spark_scale_activation.py'
header=EXPORT/'python/sglang/kernels/jit/csrc/elementwise/muse_spark_scale_activation.cuh'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(header)=='7a107cd80006c384f104294c7b835af0b9349b4fcd3c4cc9f52541e9c7304389'
name='sglang.kernels.ops.moe.muse_spark_scale_activation';spec=importlib.util.spec_from_file_location(name,wrapper);fused=importlib.util.module_from_spec(spec);sys.modules[name]=fused;spec.loader.exec_module(fused)
real_load_jit=fused.load_jit
relocations=[]
def relocated_load_jit(*args,**kwargs):
    assert kwargs['cuda_files']==['elementwise/muse_spark_scale_activation.cuh']
    kwargs['cuda_files']=[str(header)]
    relocations.append({'module_args':[str(x) for x in args],'header_sha256':sha(header),'replacement_cuda_files':kwargs['cuda_files']})
    return real_load_jit(*args,**kwargs)
fused.load_jit=relocated_load_jit
reference=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py'
reference_name='sglang.kernels.ops.moe.muse_spark_humming_scale';spec=importlib.util.spec_from_file_location(reference_name,reference);ref=importlib.util.module_from_spec(spec);sys.modules[reference_name]=ref;spec.loader.exec_module(ref)
manifest=json.loads((P/'scale_activation_export/manifest.json').read_text())
for relative,digest in manifest['sources'].items():
    original=EXPORT/relative;assert sha(original)==digest
    snapshot=a.out/'executed_export_sources'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(original.read_bytes());assert sha(snapshot)==digest
report={'status':'running','scope':'Exact exported wrapper and CUDA source, only relative CUDA filename relocated to isolated export; no arithmetic stubs or production modifications','sources':{'wrapper':sha(wrapper),'header':sha(header),'reference_scale':sha(reference),'harness':sha(Path(__file__))},'checks':[],'fallback_checks':[],'relocations':relocations}
def save():
    temp=a.out/'results.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.out/'results.json')
def baseline(x,out,ids,scales):
    scratch=x.clone();ref.scale_active_rows(scratch,ids,scales);silu_and_mul(scratch,out,expert_ids=ids,expert_step=1)
def check(actual,expected,**context):
    mismatches=int((actual.view(torch.int16)!=expected.view(torch.int16)).sum().item())
    row={'context':context,'mismatches':mismatches,'elements':actual.numel(),'passed':mismatches==0,'monotonic_seconds':time.monotonic()}
    report['checks'].append(row)
    if mismatches:report['status']='failed_bitwise'
    save();assert mismatches==0,row
for rows,width,dtype in ((8,4096,torch.int32),(33,512,torch.int64),(128,4096,torch.int32),(513,512,torch.int64)):
    torch.manual_seed(rows);x=torch.randn((rows,width*2),device='cuda',dtype=torch.bfloat16)*4
    ids=(torch.arange(rows,device='cuda')%33-1).to(dtype);scales=torch.linspace(.25,2,32,device='cuda',dtype=torch.bfloat16)
    actual=torch.full((rows,width),7,device='cuda',dtype=torch.bfloat16);expected=torch.full_like(actual,7)
    assert fused._supports_scaled_activation(x,actual,ids,scales)
    baseline(x,expected,ids,scales);fused.scaled_silu_and_mul(x,actual,ids,scales)
    check(actual,expected,rows=rows,width=width,dtype=str(dtype),mode='eager')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):fused.scaled_silu_and_mul(x,actual,ids,scales)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fused.scaled_silu_and_mul(x,actual,ids,scales)
    for mode in ('all_remote','all_local','mixed'):
        x.copy_(torch.randn_like(x)*8);scales.copy_(torch.linspace(2,.25,32,device='cuda',dtype=torch.bfloat16))
        ids.fill_(-1) if mode=='all_remote' else ids.copy_((torch.arange(rows,device='cuda')%(32 if mode=='all_local' else 33)-(0 if mode=='all_local' else 1)).to(dtype))
        actual.fill_(7);expected.fill_(7);baseline(x,expected,ids,scales);graph.replay()
        check(actual,expected,rows=rows,width=width,dtype=str(dtype),mode=mode)
    del graph
# Unsupported layouts/dtypes retain the original pair's error contract.
for mode in ('noncontiguous','float16'):
    x=torch.zeros((2,1024),device='cuda',dtype=torch.bfloat16);out=torch.empty((2,512),device='cuda',dtype=torch.bfloat16)
    ids=torch.zeros(2,device='cuda',dtype=torch.int32);scales=torch.ones(32,device='cuda',dtype=torch.bfloat16)
    if mode=='noncontiguous':x=torch.zeros((2,2048),device='cuda',dtype=torch.bfloat16)[:,::2]
    else:x=x.to(torch.float16);out=out.to(torch.float16)
    assert not fused._supports_scaled_activation(x,out,ids,scales)
    errors=[]
    for fn in (lambda:ref.scale_active_rows(x,ids,scales),lambda:fused.scaled_silu_and_mul(x,out,ids,scales)):
        try:fn();errors.append(None)
        except Exception as error:errors.append({'type':type(error).__name__,'message':str(error)})
    passed=errors[0] is not None and errors[0]==errors[1]
    report['fallback_checks'].append({'mode':mode,'errors':errors,'passed':passed})
    if not passed:report['status']='failed_fallback_gate'
    save();assert passed
report.update(status='completed',passed=True,source_unchanged=report['sources']['wrapper']==sha(wrapper) and report['sources']['header']==sha(header));save();assert report['source_unchanged']
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'fallback_checks':len(report['fallback_checks'])}),flush=True)