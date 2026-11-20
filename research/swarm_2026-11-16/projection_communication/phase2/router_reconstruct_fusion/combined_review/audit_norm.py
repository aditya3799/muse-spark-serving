"""Execute exact candidate caller and guards with CPU spies; no arithmetic claim."""
from pathlib import Path
import ast,contextlib,hashlib,json,types
P=Path(__file__).resolve().parent
source=Path('D:\\OpenKernelChallenge\\serving\\swarm_2026-11-16\\coordination\\research_resumed\\norm_glue_router\\candidate_sources\\python\\sglang\\srt\\models\\muse_spark.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='c01be0b1b3b7091ad8c31779c962ed92e036d63b8f3d32fbb321ec8adc0db676'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xMoE')
names=['_forward_experts','_can_shard_prefill_row_projection','_can_shard_small_projection']
nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
S=types.SimpleNamespace
class Tensor:
 dtype='bf16';device='fake'
 def __init__(self,m=512):self.shape=(m,8192)
 def float(self):return self
class MuseSpark1xRMSNorm:
 def __init__(self):self.calls=0
 def __call__(self,x):self.calls+=1;return x
class SubNorm(MuseSpark1xRMSNorm):pass
class OtherNorm(MuseSpark1xRMSNorm):pass
rows=[]
for norm_kind in ('exact','subclass','custom','absent'):
 for path in ('row_nvfp4','row_generic','small','unsupported','sp','dp','cp','deterministic','nonmarlin','no_projection','prepare_early'):
  state=S(sp_active=path=='sp');events=[];norm={'exact':MuseSpark1xRMSNorm,'subclass':SubNorm,'custom':OtherNorm,'absent':lambda:None}[norm_kind]()
  x=Tensor(64 if path in ('small','prepare_early') else 512)
  par=S(attn_dp_size=2 if path=='dp' else 1,attn_cp_size=2 if path=='cp' else 1,tp_group=S(ca_comm=None))
  class Experts:
   use_marlin=path not in ('nonmarlin','prepare_early');post_expert_norm=None
   def prepare_input(self,x,n):
    events.append(('prepare_norm',n is norm));return n(x) if n is not None else x
   def __call__(self,x,*args,**kwargs):
    n=kwargs.get('pre_expert_norm');events.append(('runner_norm',n is norm,n is None))
    if n is not None and kwargs.get('prepared_input') is None:n(x)
    return x
  def row(g,x,w,*,row_norm=None):
   events.append(('row',row_norm is norm,row_norm is None));return row_norm(x) if row_norm is not None else x
  def project(*args):events.append(('regular_projection',));return x
  def sp_forward(x,*args):
   events.append(('sp',));return norm(x) if norm is not None else x
  ns={'torch':S(Tensor=Tensor,float32='fp32',bfloat16='bf16',float16='fp16',cuda=S(is_available=lambda:False,get_device_capability=lambda d:(9,0))), 'Optional':__import__('typing').Optional,'contextlib':contextlib,'_is_cpu':False,'_use_split_bf16_router':lambda m:True,'get_forward':lambda:state,'get_parallel':lambda:par,'envs':S(SGLANG_ENABLE_DETERMINISTIC_INFERENCE=S(get=lambda:path=='deterministic')),'prefill_row_projection_supported':lambda g,x,w:path not in ('small','unsupported','prepare_early'),'small_projection_supported':lambda g,x,w:path=='small','gemm_ag_supported':lambda *a:False,'MuseSpark1xRMSNorm':MuseSpark1xRMSNorm,'prefill_row_projection':row,'small_projection':project,'bf16_gemm_dispatch':project,'get_global_experts_capturer':lambda:None}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
  owner=S(router_in_fp32=False,nvfp4_checkpoint=path!='row_generic',pre_expert_norm=norm,pre_expert_proj=None if path=='no_projection' else S(weight=object()),experts=Experts(),tp_size=8,expert_hidden_size=4096,top_k=8,gate=lambda x:x,_routing_op=lambda:lambda *a,**k:(x,x,x),_forward_sp_experts=sp_forward)
  for name in names:setattr(owner,name,types.MethodType(ns[name],owner))
  assert owner._forward_experts(x) is x
  assert owner.pre_expert_norm is norm
  expected_row=path in ('row_nvfp4','row_generic')
  local=expected_row and norm_kind=='exact'
  assert sum(e[0]=='row' for e in events)==int(expected_row),(path,events)
  assert (norm.calls if norm is not None else 0)==int(norm is not None),(path,norm_kind,events)
  if expected_row:assert next(e for e in events if e[0]=='row')[2]==(not local)
  if path!='sp' and path!='row_generic':
   runner=next(e for e in events if e[0]=='runner_norm');assert runner[2]==(local or norm is None),(path,norm_kind,events)
  rows.append({'path':path,'norm_kind':norm_kind,'norm_calls':norm.calls if norm is not None else 0,'events':events,'passed':True})
report={'passed':True,'case_count':len(rows),'source_sha256':sha(source),'harness_sha256':sha(Path(__file__)),'scope':'Exact extracted _forward_experts plus both eligibility methods executed with CPU spies; norm arithmetic and CUDA/collective behavior not tested here. SP expert body is a norm-count spy.','cases':rows}
(P/'CPU_CALLER_COMPOSITION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(rows),'source_sha256':sha(source)}))
