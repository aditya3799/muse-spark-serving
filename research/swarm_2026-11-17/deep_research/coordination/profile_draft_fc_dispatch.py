"""Identify native GEMM dispatch/reduction differences for draft FC sharding."""
import argparse,json,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent

def main():
 import torch
 import torch.nn.functional as F
 from torch.profiler import profile,ProfilerActivity
 parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1);torch.manual_seed(617);torch.backends.cuda.matmul.allow_tf32=False
 initial=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
 weight=torch.randn((8192,40960),device='cuda',dtype=torch.bfloat16)*.003
 results=[]
 def compare(a,b):
  delta=a.float()-b.float()
  return {'bitwise':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'different_elements':int((a!=b).sum()),'max_abs':float(delta.abs().max()),'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.float()).clamp_min(1e-30))}
 for m in (8,128,768,4096):
  x=torch.randn((m,40960),device='cuda',dtype=torch.bfloat16)
  inherited_output=None
  for flag in (True,False):
   torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=flag
   outputs={}
   for name,fn in [('base',lambda:F.linear(x,weight)),('column_rank0',lambda:F.linear(x,weight[:1024])),('row_rank0',lambda:F.linear(x[:(m+7)//8],weight))]:
    for _ in range(2):fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
     result=fn();torch.cuda.synchronize()
    outputs[name]=result
    trace=out/f'm{m}_reduce{int(flag)}_{name}.json';prof.export_chrome_trace(str(trace));events=json.loads(trace.read_text())['traceEvents'];kernels=[{'name':e['name'],'duration_us':e.get('dur'),'args':e.get('args')} for e in events if e.get('cat')=='kernel']
    results.append({'rows':m,'reduced_precision_allowed':flag,'operation':name,'kernels':kernels,'trace':trace.name,'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest()})
   checks={'column_rank0':compare(outputs['base'][:,:1024].contiguous(),outputs['column_rank0']),'row_rank0':compare(outputs['base'][:(m+7)//8],outputs['row_rank0'])}
   if flag:inherited_output=outputs['base']
   else:checks['full_vs_inherited_baseline']=compare(inherited_output,outputs['base'])
   results.append({'rows':m,'reduced_precision_allowed':flag,'checks':checks})
   print(json.dumps({'rows':m,'flag':flag,'checks':checks}),flush=True)
  del x,outputs,inherited_output
  torch.cuda.empty_cache()
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=initial
 report={'status':'LOCAL_NATIVE_GEMM_DISPATCH_DIAGNOSTIC','device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'torch':torch.__version__,'initial_reduction_flag':initial,'geometry':[8192,40960],'results':results,'limits':['Synthetic draft FC weights and features','Profiling durations are instrumented and not accepted benchmark timings','Local CUDA/cuBLAS version may differ from judge image','Disabling reduced-precision reduction is diagnostic, not an enabled candidate change']}
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
