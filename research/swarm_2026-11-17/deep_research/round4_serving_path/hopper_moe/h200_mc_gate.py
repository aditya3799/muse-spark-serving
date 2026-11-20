"""Future isolated eight-H200 numeric gate. Default --preflight never imports CUDA."""
import argparse,hashlib,json,os,pathlib,sys
P=pathlib.Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
def hashes():return {n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['mc_workspace_handoff.cuh','mc_ready_plane_handoff.cuh','h200_mc_gate.cuh','h200_mc_gate.py']}
def preflight():
 ledger=json.loads((P/'SOURCE_COLLECTIVE_AUDIT.json').read_text())
 for rel,expected in ledger['source_hashes'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
 return dict(status='STATIC_PREFLIGHT_ONLY_NATIVE_EXECUTION_HELD',sources=hashes(),scope='attention o_proj→MC2shot→actualpostweightFalse/routerFalse norm; no full model',requires='8 local H200 GPUs; explicit --run-h200; future authorization',unsupported=['SM86/H100/Blackwell','world!=8','DP/SP','GEMV/custom/deterministic GEMM','unicast/push/NCCL algorithms','concurrent users of same communicator plane','retaining borrowed AR output after norm'],PDL='candidate disablesPDL; compare against stock and matchedPDLoff numerically; no speed claim')
def run(outdir):
 sys.path.insert(0,str(ROOT/'sglang/python'))
 import torch,torch.distributed as dist
 import torch.nn.functional as F
 from datetime import timedelta
 assert os.environ.get('WORLD_SIZE')=='8' and os.environ.get('LOCAL_WORLD_SIZE')=='8'
 assert os.environ.get('MASTER_ADDR','127.0.0.1') in ['localhost','127.0.0.1','::1'],'single-node fixture only'
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
 assert torch.cuda.get_device_capability()==(9,0) and 'H200' in torch.cuda.get_device_name().upper()
 assert not torch.are_deterministic_algorithms_enabled()
 from sglang.kernels.jit.utils import load_jit
 from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import CustomAllReduceV2
 from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
 from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import fused_norm_residual as stock_norm
 for module_name in ['sglang.srt.distributed.device_communicators.custom_all_reduce_v2','sglang.kernels.ops.communication.all_reduce','sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit']:
  imported=pathlib.Path(sys.modules[module_name].__file__).resolve()
  expected=(ROOT/'sglang/python'/pathlib.Path(*module_name.split('.'))).with_suffix('.py').resolve()
  assert imported==expected,'imported unpinned source: '+str(imported)
 dist.init_process_group('nccl',timeout=timedelta(seconds=180));cpu=dist.new_group(backend='gloo',timeout=timedelta(seconds=180))
 comm=CustomAllReduceV2(group=cpu,device=torch.device('cuda',rank))
 def agree(ok,label):
  flag=torch.tensor(int(bool(ok)),dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MIN,group=cpu);assert flag.item()==1,label
 agree(not comm.disabled and comm.world_size==8 and comm.has_multicast and comm.pull_enabled,'V2 MC unavailable')
 module=load_jit('research_mc_handoff_gate',hashes()['h200_mc_gate.cuh'],cuda_files=[str(P/'h200_mc_gate.cuh')],extra_include_paths=[str(P),str(ROOT/'sglang/python/sglang/kernels/jit/csrc/distributed'),str(ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark')],cuda_wrappers=[('local_address','research_local_pull_address'),('staged','ResearchMCWorkspaceHandoff::run'),('direct','ResearchMCReadyPlaneHandoff::run'),('control','AllReduceKernel<bf16_t,8,false>::run'),('norm_small','MuseSparkNormResidual<8192,256,false,true,1,false,false>::run'),('norm_large','MuseSparkNormResidual<8192,512,false,false,1,false,false>::run')])
 outdir.mkdir(parents=True,exist_ok=True);report=preflight();report.update(status='NATIVE_GATE_RUNNING',rank=rank,checks=[])
 def save():(outdir/('rank%d.json'%rank)).write_text(json.dumps(report,indent=2))
 def tensor(x):return x if isinstance(x,torch.Tensor) else torch.from_dlpack(x)
 def check(x,y,label):
  ok=x.dtype==y.dtype and x.shape==y.shape and torch.isfinite(x).all().item() and torch.isfinite(y).all().item() and torch.equal(x.view(torch.int16 if x.dtype==torch.bfloat16 else torch.int32),y.view(torch.int16 if y.dtype==torch.bfloat16 else torch.int32))
  agree(ok,label);report['checks'].append(dict(label=label,finite_bitwise=ok));save()
 for m in [32,128,256,753]:
  nbytes=m*8192*2;cfg=comm._pick_config(nbytes,False)
  agree(comm.override_algo is None and cfg is not None and cfg.algo==AllReduceAlgo.TWO_SHOT_PULL and cfg.use_multicast and not cfg.use_graph,'not stock MC algorithm')
  offset=2*8*comm.max_push_size
  view=comm._symm_tensor.view(torch.uint8)[offset:offset+nbytes].view(torch.bfloat16).view(m,8192)
  agree(view.data_ptr()==module.local_address(comm.obj),'local ordinaryVA pointer mismatch')
  x=torch.empty((m,1024),device='cuda',dtype=torch.bfloat16);w=torch.empty((8192,1024),device='cuda',dtype=torch.bfloat16);stream=torch.empty((m,8192),device='cuda',dtype=torch.float32)
  pre=torch.empty(8192,device='cuda',dtype=torch.bfloat16);alpha=torch.empty(8192,device='cuda',dtype=torch.float32);beta=torch.empty_like(alpha)
  normfn=module.norm_small if m<192 else module.norm_large
  def norm(branch):
   out=torch.full_like(branch,float('nan'));res=torch.full_like(stream,float('nan'))
   normfn(branch,stream,out,res,None,None,pre,alpha,beta,1e-8,1e-5,False)
   return out,res
  for gen in range(3):
   torch.manual_seed(18700+gen+rank*111);x.normal_();w.normal_(0,.125)
   # Residual and gate/normalization parameters must be replicated, as in TP target.
   torch.manual_seed(18800+gen);stream.normal_();pre.normal_(1,.1);g=torch.randn_like(alpha)/.3;beta.copy_(torch.sigmoid(g));alpha.copy_(torch.sqrt(torch.clamp(torch.sigmoid(-g)*(1+beta),min=1e-3)))
   partial=F.linear(x,w,None);reference=tensor(comm.custom_all_reduce(partial));off_norm=norm(reference)
   stock_out=torch.full_like(reference,float('nan'));stock_res=torch.full_like(stream,float('nan'))
   ref_norm=stock_norm(reference,stream,1e-8,1e-5,post_norm_is_input_scale=False,post_weight=None,pre_weight=pre,alpha=alpha,beta=beta,want_router=False,pre_norm_bf16=False,output=stock_out,new_stream_out=stock_res)[:2]
   for label,a,b in zip(['output','stream'],ref_norm,off_norm):check(a,b,'M%dG%d stock_vs_PDLoff_norm_%s'%(m,gen,label))
   control=tensor(module.control(comm.obj,partial,'2shot_pull',None,True));check(reference,control,'M%dG%d stock_vs_PDLoff_AR'%(m,gen))
   borrowed=tensor(module.staged(comm.obj,partial,view));raw=borrowed.clone();candidate=norm(borrowed)
   check(reference,raw,'M%dG%d staged_AR'%(m,gen))
   for label,a,b in zip(['output','stream'],ref_norm,candidate):check(a,b,'M%dG%d staged_%s'%(m,gen,label))
   torch.mm(x,w.t(),out=view);check(partial,view,'M%dG%d producer'%(m,gen))
   borrowed=tensor(module.direct(comm.obj,view,view));raw=borrowed.clone();candidate=norm(borrowed)
   check(reference,raw,'M%dG%d direct_AR'%(m,gen))
   for label,a,b in zip(['output','stream'],ref_norm,candidate):check(a,b,'M%dG%d direct_%s'%(m,gen,label))
   # All checks finish before any following collective can borrow this plane.
   torch.cuda.synchronize();dist.barrier(group=cpu)
 report.update(status='NATIVE_EAGER_NUMERIC_PASS_CAPTURE_AND_TIMING_PENDING',no_serving_claim=True);save();dist.destroy_process_group()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-h200',action='store_true');ap.add_argument('--out',default=str(P/'future_h200_mc_gate'));a=ap.parse_args()
 print(json.dumps(preflight(),indent=2))
 if a.run_h200:run(pathlib.Path(a.out))
