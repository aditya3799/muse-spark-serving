"""Pinned-source SM90 metadata audit; architecture interfaces mocked, no kernels."""
from pathlib import Path
import sys,types,importlib.util,json,hashlib
p=Path(__file__).resolve().parent;vendor=p.parent/'moe/humming_vendor/humming'
for name in ['humming','humming.config','humming.tune','humming.utils']:
 mod=types.ModuleType(name);mod.__path__=[];sys.modules[name]=mod
fake=types.ModuleType('torch');fake.cuda=types.SimpleNamespace(get_device_capability=lambda:(9,0))
for name in ['float8_e8m0fnu','float8_e4m3fn','float8_e5m2','float16','bfloat16','float32']:
 setattr(fake,name,'torch.'+name)
sys.modules['torch']=fake
hashes={}
def load(name,rel):
 path=vendor/rel;hashes[rel]=hashlib.sha256(path.read_bytes()).hexdigest()
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
 sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
d=load('humming.dtypes','dtypes.py');sys.modules['humming'].dtypes=d
enums=load('humming.config.enum','config/enum.py')
load('humming.config.base','config/base.py')
config=load('humming.config.config','config/config.py')
for name in ['LayerConfig','ComputeConfig','TuningConfig']:
 setattr(sys.modules['humming.config'],name,getattr(config,name))
for name in ['GemmType','MmaType','WeightScaleType','WeightScale2Type']:
 setattr(sys.modules['humming.config'],name,getattr(enums,name))
def unused(*args,**kwargs):raise AssertionError('Unexpected hardware-dependent helper in SM90 heuristic audit')
for name,attr in [('humming.utils.device','get_device_num_sms'),('humming.utils.smem','estimate_smem_size_layer')]:
 mod=types.ModuleType(name);setattr(mod,attr,unused);sys.modules[name]=mod
load('humming.tune.base','tune/base.py');sm90=load('humming.tune.sm90','tune/sm90.py')
import numpy as np
from ep_tuning_selector import select_ep_config
reference=json.loads((p.parent/'moe/sm90_wgmma_metadata_audit.json').read_text())
intervals={}
for experts in (4,8,16,32,64):
 intervals[experts]={}
 for stage,n in [('w13',8192),('w2',4096)]:
  meta=config.LayerConfig(shape_n=n,shape_k=4096,num_experts=experts,a_dtype=d.bfloat16,b_dtype=d.float4e2m1,c_dtype=d.bfloat16,bs_dtype=d.float8e4m3,weight_scale_group_size=16,weight_scale_type=enums.WeightScaleType.GROUP,weight_scale_2_type=enums.WeightScale2Type.NONE)
  iv=sm90.Sm90Heuristics.get_configs(meta,use_f16_accum=False,gemm_type=enums.GemmType.INDEXED)
  for lo,hi,c in iv:c['use_stream_k']=False
  intervals[experts][stage]=iv
  if experts==32:assert json.loads(json.dumps(iv))==reference['stages'][stage]['intervals']
 print('Generated intervals for localE',experts,flush=True)

def routes(m,topk,g,l,pattern,seed):
 rng=np.random.default_rng(seed)
 starts=rng.integers(0,g,size=(m,1));strides=2*rng.integers(0,g//2,size=(m,1))+1
 ids=(starts+strides*np.arange(topk))%g
 if pattern=='rank_concentrated':
  take=min(topk,l)
  ids[:,:take]=(rng.integers(0,l,size=(m,1))+np.arange(take))%l
  if take<topk:ids[:,take:]=l+np.arange(topk-take)
 elif pattern=='hot_expert':
  full=(starts+strides*np.arange(topk+1))%g
  ids=np.empty((m,topk),dtype=np.int64);ids[:,0]=0
  for row in range(m):ids[row,1:]=full[row][full[row]!=0][:topk-1]
 elif pattern=='all_remote':
  if g-l<topk:return None
  for row in range(m):ids[row]=l+rng.choice(g-l,size=topk,replace=False)
 assert ids.min()>=0 and ids.max()<g
 assert all(len(set(row))==topk for row in ids)
 return ids

def work(counts,cfg,n,k):
 bm,bn,bk=cfg['block_shape'];blocks=int(((counts+bm-1)//bm).sum());padded=blocks*bm
 return dict(block_shape=list(cfg['block_shape']),active_routes=int(counts.sum()),padded_rows=padded,padding_rows=padded-int(counts.sum()),row_blocks=blocks,logical_output_tiles=blocks*((n+bn-1)//bn),k_iterations=(k+bk-1)//bk)

rows=[]
for l in intervals:
 for world in (1,2,4,8):
  g=l*world
  for topk in (1,4,8,16):
   if topk>g:continue
   for m in (1,8,16,32,33,64,128,256,512,1024,4096):
    actual=m*topk
    chosen={s:(next(c for lo,hi,c in iv if lo<actual<=hi),select_ep_config(iv,actual,l,g)) for s,iv in intervals[l].items()}
    for seed in (17,41,73):
     for pattern in ('uniform','rank_concentrated','hot_expert','all_remote'):
      ids=routes(m,topk,g,l,pattern,seed)
      if ids is None:continue
      counts=np.bincount(ids.ravel(),minlength=g);assert int(counts.sum())==actual
      for rank in (range(world) if pattern!='all_remote' else (0,)):
       local=counts[rank*l:(rank+1)*l]
       for stage,(old,new) in chosen.items():
        assert new['valid_shape_m']==actual
        n=8192 if stage=='w13' else 4096
        rows.append(dict(local_experts=l,global_experts=g,world_size=world,topk=topk,m=m,seed=seed,pattern=pattern,rank=rank,stage=stage,actual_routes=actual,tune_m=new['tune_m'],before=work(local,old,n,4096),after=work(local,new['config'],n,4096)))
 print('Analyzed routes localE',l,flush=True)
aggregates=[]
for pattern in ('uniform','rank_concentrated','hot_expert','all_remote'):
 for stage in ('w13','w2'):
  rr=[r for r in rows if r['pattern']==pattern and r['stage']==stage and r['world_size']>1]
  totals={side:{key:sum(r[side][key] for r in rr) for key in ('active_routes','padded_rows','padding_rows','row_blocks','logical_output_tiles')} for side in ('before','after')}
  aggregates.append(dict(pattern=pattern,stage=stage,cases=len(rr),totals=totals,padding_improved=sum(r['after']['padding_rows']<r['before']['padding_rows'] for r in rr),padding_worse=sum(r['after']['padding_rows']>r['before']['padding_rows'] for r in rr),block_count_increased=sum(r['after']['row_blocks']>r['before']['row_blocks'] for r in rr)))
worst=sorted((r for r in rows if r['before']['row_blocks']>0),key=lambda r:r['after']['row_blocks']/r['before']['row_blocks'],reverse=True)[:20]
examples=[r for r in rows if r['local_experts']==32 and r['global_experts']==256 and r['topk']==8 and r['m'] in (16,32,128,256,1024) and r['seed']==17 and r['rank']==0]
report=dict(scope='CPU synthetic routing and exact pinned SM90 heuristics only; no kernel or runtime speed prediction.',source_hashes=hashes,reference_E32_intervals_exact=True,selector_policy='tune_m=ceil(actual_routes*localE/globalE); valid_shape_m remains actual_routes',synthetic_uniform='Uniform route marginals and distinct top-k per token from random cyclic starts/odd strides. Structured correlations; not independent subset sampling.',cases=len(rows),aggregates=aggregates,examples=examples,worst_row_block_ratio_cases=worst,all_remote_note='Zero active rows yield zero logical work in both variants; launches and synchronization are not modeled.',intervals=intervals)
(p/'ep_tuning_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(p/'ep_tuning_cases.json').write_text(json.dumps(rows,separators=(',',':')),encoding='utf-8')
print(json.dumps(dict(cases=len(rows),aggregates=aggregates),indent=2))
