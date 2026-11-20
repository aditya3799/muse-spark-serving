"""Deterministic saved-row normalization/cast diagnostic; no performance claims."""
import hashlib,json,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;D=H.parent/'decode';OUT=H/'prologue_rounding_debug'
sys.path[:0]=[str(D),str(OUT)]
import torch,triton
from component_fixture import native_norm,_triton_config
from prologue_package.kernel import fusion18_kernel
from prologue_package import Fa3Prologue
from native_debug import native_debug_kernel
from candidate_debug import candidate_debug_kernel

@torch.no_grad()
def main():
 torch.set_num_threads(2);started=time.perf_counter()
 payload=json.loads((H/'prefill_prologue_diagnostic.json').read_text())['diagnostics']['samples'][0]['input_head']
 row=torch.tensor(payload,device='cuda',dtype=torch.bfloat16)
 adapter=Fa3Prologue();codec=adapter.prepare_codec(torch.device('cuda',0));assert codec=='software'
 native=native_norm.__globals__['_qk_rms_norm_fwd_kernel']
 cases=[]
 for n,padded,token in [(4095,4096,2),(16383,16384,4546)]:
  x=torch.zeros((padded,2304),device='cuda',dtype=torch.bfloat16)
  x[token,:64].copy_(row);x[token,1024:1088].copy_(row)
  loc=torch.arange(1,n+1,device='cuda',dtype=torch.int64);cap=n+1
  nq=torch.empty((padded,1024),device='cuda',dtype=x.dtype);nk=torch.empty((padded,128),device='cuda',dtype=x.dtype)
  cq=torch.empty((n,1024),device='cuda',dtype=torch.uint8)
  ck,cv=[torch.zeros((cap,128),device='cuda',dtype=torch.uint8) for _ in range(2)]
  native_args=[x,nq,nk,x,x,x,x,x,1e-5,2304,padded*18,0,16,2,64,64,_triton_config(padded*18)[0]]
  native_kw=dict(HAS_ROPE=False,HAS_KV_STORE=False,FP8_KV_STORE=False,num_warps=4)
  native_compiled=native[(triton.cdiv(padded*18,_triton_config(padded*18)[0]),)](*native_args,**native_kw)
  candidate_args=[x,cq,x,x,x,ck,cv,loc,1e-5,2304,n*18,cap,16,2,64,64,adapter._block(n),False,True,True,False]
  candidate_kw=dict(NATIVE=False,num_warps=8,debug=True)
  cand_compiled=fusion18_kernel[(triton.cdiv(n*18,adapter._block(n)),)](*candidate_args,**candidate_kw)
  torch.cuda.synchronize()
  base=dict(q_equal=torch.equal(cq,nq[:n].to(torch.float8_e4m3fn).view(torch.uint8)),
    k_equal=torch.equal(ck[1:],nk[:n].to(torch.float8_e4m3fn).view(torch.uint8)),
    native_bf16=nk[token,:64].float().cpu().tolist(),native_fp8=nk[token,:64].to(torch.float8_e4m3fn).view(torch.uint8).cpu().tolist(),
    candidate_fp8=ck[token+1,:64].cpu().tolist())
  dbg={}
  for label,kernel,args,kwargs,count in [('native',native_debug_kernel,native_args,native_kw,padded),('candidate',candidate_debug_kernel,candidate_args,candidate_kw,n)]:
   dx=torch.empty((count,128),device='cuda',dtype=torch.float32)
   dv,dr=[torch.empty((count,2),device='cuda',dtype=torch.float32) for _ in range(2)]
   db=torch.empty((count,128),device='cuda',dtype=torch.bfloat16)
   args=list(args);saved_q=args[1].clone();saved_k=(args[2] if label=='native' else args[5]).clone()
   args[1]=torch.empty_like(args[1])
   if label=='native':args[2]=torch.empty_like(args[2])
   else:args[5]=torch.zeros_like(args[5]);args[6]=torch.zeros_like(args[6])
   args[8:8]=[dx,dv,dr,db]
   block=_triton_config(padded*18)[0] if label=='native' else adapter._block(n)
   compiled=kernel[(triton.cdiv(count*18,block),)](*args,**kwargs)
   torch.cuda.synchronize()
   final_k=args[2] if label=='native' else args[5]
   dbg[label]=dict(variance=float(dv[token,0]),rstd=float(dr[token,0]),
      pre_cast=dx[token,:64].cpu().tolist(),rounded_bf16=db[token,:64].float().cpu().tolist(),
      debug_q_matches_original=torch.equal(args[1],saved_q),debug_k_matches_original=torch.equal(final_k,saved_k))
   for name in ('ptx','ttgir'):
    if name in compiled.asm:(OUT/f'warp8_{label}_debug_m{n}.{name}').write_text(compiled.asm[name])
   del dx,dv,dr,db,saved_q,saved_k,compiled
  for label,compiled in [('native',native_compiled),('candidate',cand_compiled)]:
   for name in ('ptx','ttgir'):
    if name in compiled.asm:(OUT/f'warp8_{label}_original_m{n}.{name}').write_text(compiled.asm[name])
  cases.append(dict(real=n,padded=padded,token=token,native_block=_triton_config(padded*18)[0],candidate_block=adapter._block(n),original=base,debug=dbg))
  print(n,'Q',base['q_equal'],'K',base['k_equal'],'var',dbg['native']['variance'],dbg['candidate']['variance'],'rstd',dbg['native']['rstd'],dbg['candidate']['rstd'],flush=True)
  del x,nq,nk,cq,ck,cv,args,native_args,candidate_args
 doc=dict(scope=__doc__,device=torch.cuda.get_device_name(),codec=codec,cases=cases,
   elapsed_s=time.perf_counter()-started,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
   instrumentation_warning='Added stores may change compilation. Final debug outputs are explicitly compared to originals; only matching-instrumentation values support localization.',
   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (OUT/'warp8_reproducer.json').write_text(json.dumps(doc,indent=2))
 print('COMPLETE',doc['elapsed_s'],doc['peak_allocated'],doc['peak_reserved'])
if __name__=='__main__':main()

