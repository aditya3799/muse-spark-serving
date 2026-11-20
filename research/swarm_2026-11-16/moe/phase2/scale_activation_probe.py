"""Bitwise scale/SiLU semantics and pair timing; no serving performance claim."""
import argparse, hashlib, importlib.util, json, statistics, sys, time
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--sglang-root',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
from kernel_runtime import bootstrap
provenance=bootstrap(a.sglang_root)
import torch
from scale_activation import scaled_silu_and_mul
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.jit.utils import get_activation_cuda_cflags
source=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='1fb726a24a8a3335f6b894b602223ca252783607594aea0d108e3c01637bbddd'
spec=importlib.util.spec_from_file_location('scored_scale_reference',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
report={'status':'running','scope':'Exact bitwise pair component probe; no GEMM, H200 or serving claim',
 'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda_flags':get_activation_cuda_cflags(),
 'source_hashes':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('scale_activation.py','scale_activation.cuh','scale_activation_probe.py')},
 'stock_activation_sha256':hashlib.sha256((a.sglang_root/'python/sglang/kernels/jit/csrc/elementwise/activation.cuh').read_bytes()).hexdigest(),
 'checks':[],'timings':[]}
def save():
    path=a.out/'results.tmp';path.write_text(json.dumps(report,indent=2)+'\n');path.replace(a.out/'results.json')
def check(name,actual,expected,**context):
    bits=actual.view(torch.int16)!=expected.view(torch.int16)
    mismatch=int(bits.sum().item());finite=torch.isfinite(actual)&torch.isfinite(expected)
    max_abs=float((actual[finite].float()-expected[finite].float()).abs().max().item()) if bool(finite.any()) else None
    row={'name':name,'context':context,'mismatches':mismatch,'elements':actual.numel(),'finite_max_abs':max_abs,'passed':mismatch==0,'unix_seconds':time.time(),'monotonic_seconds':time.monotonic()}
    if mismatch:
        ix=int(bits.flatten().nonzero()[0]);row.update(first_flat_index=ix,actual_bits=int(actual.view(torch.int16).flatten()[ix]),expected_bits=int(expected.view(torch.int16).flatten()[ix]))
    report['checks'].append(row)
    if mismatch:report['status']='failed_bitwise_gate'
    save();assert mismatch==0,row

def baseline(x,ids,scales,out):
    scratch=x.clone();ref.scale_active_rows(scratch,ids,scales)
    return silu_and_mul(scratch,out,expert_ids=ids,expert_step=1)
# Every finite BF16 bit pattern appears as both gate and up across paired halves.
bits=torch.arange(65536,dtype=torch.int32,device='cuda');bits=bits[(bits&0x7f80)!=0x7f80]
values=bits.to(torch.int16).view(torch.bfloat16).reshape(-1,256)
x=torch.cat((values,values.flip(0,1)),dim=1).contiguous();ids=torch.zeros(x.shape[0],device='cuda',dtype=torch.int32)
for scale in (0.,1.,-1.,.5,2.,.25,1.5,16.,1e-38,1e-40,1e38,-.5):
    scales=torch.full((1,),scale,dtype=torch.bfloat16,device='cuda')
    expected=torch.full_like(values,7);actual=torch.full_like(values,7)
    baseline(x,ids,scales,expected);scaled_silu_and_mul(x,actual,ids,scales)
    check('finite_bf16_exhaustive',actual,expected,scale=scale)
for width in (512,4096):
  for rows in (1,8,64,264,1024):
    for dtype in (torch.int32,torch.int64):
      torch.manual_seed(342+rows);x=torch.randn((rows,width*2),dtype=torch.bfloat16,device='cuda')*8
      ids=(torch.arange(rows,device='cuda')%33-1).to(dtype)
      scales=torch.linspace(.25,2,32,device='cuda',dtype=torch.bfloat16)
      expected=torch.full((rows,width),7,dtype=torch.bfloat16,device='cuda');actual=torch.full_like(expected,7)
      baseline(x,ids,scales,expected);scaled_silu_and_mul(x,actual,ids,scales)
      check('mixed_routes',actual,expected,rows=rows,width=width,id_dtype=str(dtype))
      # Same graph addresses, changed inputs/scales and inactive/local route masks.
      stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):scaled_silu_and_mul(x,actual,ids,scales)
      torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):scaled_silu_and_mul(x,actual,ids,scales)
      for mode in ('all_remote','all_local','mixed'):
        x.copy_(torch.randn_like(x)*4);scales.copy_(torch.linspace(2,.25,32,device='cuda',dtype=torch.bfloat16))
        ids.fill_(-1) if mode=='all_remote' else ids.copy_((torch.arange(rows,device='cuda')%(32 if mode=='all_local' else 33)-(0 if mode=='all_local' else 1)).to(dtype))
        actual.fill_(7);expected.fill_(7);baseline(x,ids,scales,expected);graph.replay()
        check('graph_changed',actual,expected,rows=rows,width=width,id_dtype=str(dtype),mode=mode)
      del graph
# Timing uses unit scales so repeated in-place baseline bodies do not change input.
# This is a pair throughput calibration; full-chain benchmark must follow to promote.
from marlin_bench import event_sample
for rows in (8,64,264,1024):
  width=4096;x=torch.randn((rows,2*width),device='cuda',dtype=torch.bfloat16)
  ids=(torch.arange(rows,device='cuda')%33-1).to(torch.int32);scales=torch.ones(32,device='cuda',dtype=torch.bfloat16)
  out_a=torch.empty((rows,width),device='cuda',dtype=torch.bfloat16);out_b=torch.empty_like(out_a)
  def first():ref.scale_active_rows(x,ids,scales);silu_and_mul(x,out_a,expert_ids=ids,expert_step=1)
  def second():scaled_silu_and_mul(x,out_b,ids,scales)
  graphs=[]
  for fn in (first,second):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      for _ in range(8):fn()
    graphs.append(g)
  samples={key:[] for key in ('A1','B1','A2','B2')}
  for repeat in range(12):
    for label in (list(samples) if repeat%2==0 else list(reversed(samples))):
      samples[label].append(event_sample(torch,graphs[label[0]=='B'].replay,1)/8)
  med={key:statistics.median(val) for key,val in samples.items()}
  arms={arm:statistics.median(samples[arm+'1']+samples[arm+'2']) for arm in 'AB'}
  controls={arm:med[arm+'1']/med[arm+'2'] for arm in 'AB'}
  report['timings'].append({'rows':rows,'width':width,'samples_us':samples,'medians_us':arms,'speedup':arms['A']/arms['B'],'controls':controls,'noise_pass':all(abs(c-1)<=.1 for c in controls.values()),'graph_body':8,'unit_scales':True})
  save();del graphs
report.update(status='completed',passed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save()
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'timings':report['timings']}),flush=True)