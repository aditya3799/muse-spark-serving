"""Native CPU audit of producer-only policy and pinned shared-memory estimator."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,hashlib,json,math
from producer_tuning import producer_config
P=Path(__file__).resolve().parent;V=P.parent/'humming_vendor/humming'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=V/'utils/smem.py';tree=ast.parse(source.read_text(encoding='utf-8'))
class RemoveAnnotations(ast.NodeTransformer):
 def visit_FunctionDef(self,n):
  n.returns=None
  for a in n.args.args+n.args.kwonlyargs:a.annotation=None
  return self.generic_visit(n)
 def visit_AnnAssign(self,n):return ast.copy_location(ast.Assign(targets=[n.target],value=n.value),n) if n.value is not None else None
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
mod=ast.fix_missing_locations(RemoveAnnotations().visit(ast.Module(body=nodes,type_ignores=[])))
ns={'math':math,'_INT4':16,'MmaType':NS(MXMMA='mxmma'),'GemmType':NS(INDEXED='indexed',GROUPED_MASKED='grouped_masked',GROUPED_CONTIGUOUS='grouped_contiguous')}
exec(compile(mod,str(source),'exec'),ns)
meta=NS(mma_type='wgmma',a_dtype=NS(num_bits=16),b_dtype=NS(num_bits=4),bs_dtype=NS(num_bits=8),c_dtype=NS(num_bits=16),is_fp_zero_point=False,has_zero_point=False,is_channel_weight_scale=False,is_group_weight_scale=True,is_block_weight_scale=False,weight_scale_group_size=16,is_channel_weight_scale_2=False,has_bias=False,input_scale_group_size=0,num_experts=32)
audit=json.loads((P.parent/'sm90_wgmma_metadata_audit.json').read_text(encoding='utf-8'))
rows=[]
for stage,entry in audit['stages'].items():
 for lo,hi,original in entry['intervals']:
  for policy in ('baseline','tma_b','tma_b_only','warp_spec','both','warp_tma_b_only'):
   c=original if policy=='baseline' else producer_config(original,policy)
   assert c is not None,(stage,lo,hi,policy)
   ws=c.get('use_warp_spec',False);bar=c.get('use_mbarrier',False)
   smem=ns['estimate_smem_size_layer'](meta,c['block_shape'],'indexed',c['num_stages'],warp_shape=c['warp_shape'],use_mbarrier=bar,use_warp_spec=ws)
   maththreads=32
   for b,w in zip(c['block_shape'],c['warp_shape']):maththreads*=b//w
   loadthreads=128 if ws else maththreads;total=maththreads+(128 if ws else 0)
   if ws:
    loadregs=40 if maththreads>256 else 24
    numwarps=total//32;budget=65536//numwarps//256*256*numwarps
    mathregs=min(96 if maththreads>256 else 232,max(24,(budget-loadthreads*loadregs)//maththreads//8*8))
   else:loadregs=mathregs=None
   rows.append(dict(stage=stage,lo=lo,hi=hi,policy=policy,configuration=c,shared_bytes=smem,math_threads=maththreads,load_threads=loadthreads,total_threads=total,load_register_budget=loadregs,math_register_budget=mathregs,within_shared_cap=smem<=227*1024))
assert all(r['within_shared_cap'] for r in rows)
# CPU guards reject unsupportedarchitecture or precision/reduction changes.
c=audit['stages']['w2']['intervals'][0][2]
assert producer_config(c,'both',86) is None
for key in ('use_stream_k','use_f16_accum'):
 bad=dict(c);bad[key]=True;assert producer_config(bad,'both') is None
for policy in ('baseline','invalid'):assert producer_config(c,policy) is None
result={'status':'PASS_SOURCE_CPU_ONLY','scope':'Pinned geometry and shared-memory estimator, not CUDA execution or speed','source_hashes':{str(f):sha(f) for f in (source,P/'producer_tuning.py',Path(__file__),V/'tune/sm90.py',V/'testing/tuning.py',V/'include/humming/kernel/humming_ws.cuh')},'rows':rows,'guard_checks':5,'max_shared_bytes':max(r['shared_bytes'] for r in rows),'max_threads':max(r['total_threads'] for r in rows),'persistent_extra_weight_bytes':0}
(P/'producer_config_audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:result[k] for k in ('status','guard_checks','max_shared_bytes','max_threads')}));print('configurations',len(rows))
