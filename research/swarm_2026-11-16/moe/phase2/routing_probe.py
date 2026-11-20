"""Routing-only A/B against the scored source, with graph transition proof."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,statistics,sys
P=Path(__file__).resolve().parent;MOE=P.parent;ROOT=MOE.parents[1]
sys.path.insert(0,str(ROOT/'optimization_marlin_tiles_2026-11-15'))
pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--sglang-root',type=Path,required=True);pa.add_argument('--tokens',nargs='+',type=int,default=[128]);pa.add_argument('--experts',type=int,default=32);pa.add_argument('--int64',action='store_true');pa.add_argument('--four-plans',action='store_true');pa.add_argument('--samples',type=int,default=12);a=pa.parse_args();a.out.mkdir(parents=True,exist_ok=True)
if a.samples<6 or a.samples%2:raise ValueError('Even sample count >=6 required')
from kernel_runtime import bootstrap
runtime=bootstrap(a.sglang_root)
import torch
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from fused_partition_align import fused_partition_align
ref=MOE.parent/'coordination/round5_humming_adaptive_fused'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ref/'candidate.diff')=='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
base_path=ref/'candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
assert sha(base_path)=='752038f214e7f8e21e16c4af8de2c9bb43a97733e7bf61615ddbe125124d5bc8'
spec=importlib.util.spec_from_file_location('phase2_scored_split',base_path);base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
keys=(('light',8),('heavy',64),('light',16),('heavy',32)) if a.four_plans else (('light',16),('heavy',48))
cutoff=min(bm for name,bm in keys if name=='light');dtype=torch.int64 if a.int64 else torch.int32
report={'status':'running','scope':'Routing producer only on SM86; unchanged scored configurations represented by2/4 unique block-M plans; no GEMM/H200 serving speedup claim','argv':sys.argv,'harness_sha256':sha(Path(__file__)),'candidate_sha256':sha(P/'fused_partition_align.py'),'baseline_patch_sha256':sha(ref/'candidate.diff'),'baseline_split_sha256':sha(base_path),'runtime':runtime,'gpu':torch.cuda.get_device_name(),'keys':keys,'cutoff':cutoff,'experts':a.experts,'dtype':str(dtype),'rows':[],'graph_batch_replays':32}
def save():
 (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
def pattern(routes,name,seed):
 rng=random.Random(seed)
 if name=='all_remote':return [-1]*routes
 if name=='hot_rank':return [i%a.experts for i in range(routes)]
 if name=='hot_expert':return [0 if i<routes//2 else rng.randrange(-7*a.experts,a.experts) for i in range(routes)]
 if name=='threshold':return ([0]*(cutoff-1)+[1]*cutoff+[2]*(cutoff+1)+[-1]*routes)[:routes]
 return [rng.randrange(-7*a.experts,a.experts) for _ in range(routes)]
def check(result,counts,values):
 expected_counts=[values.count(e) for e in range(a.experts)]
 assert counts.cpu().tolist()==expected_counts
 for (name,bm),(slots,owners,padded) in result.items():
  n=int(padded.item());assert n%bm==0 and 0<=n<=slots.numel()
  ss=slots[:n].cpu().tolist();ee=owners[:n//bm].cpu().tolist();seen=[]
  for block,e in enumerate(ee):
   assert 0<=e<a.experts
   c=expected_counts[e];assert (c<=cutoff)==(name=='light')
   for slot in ss[block*bm:(block+1)*bm]:
    assert 0<=slot<=len(values)
    if slot<len(values):assert values[slot]==e;seen.append(slot)
  wanted=[i for i,e in enumerate(values) if 0<=e<a.experts and ((expected_counts[e]<=cutoff)==(name=='light'))]
  assert sorted(seen)==wanted and len(seen)==len(set(seen))
  wanted_padded=sum(((c+bm-1)//bm)*bm for c in expected_counts if ((c<=cutoff)==(name=='light')))
  assert n==wanted_padded
 return True

def capture(fn,repeats=1):
 for _ in range(3):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(repeats):out=fn()
 return g,out

def sample(g):
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 start.record();g.replay();end.record();end.synchronize()
 return start.elapsed_time(end)*1000/32

for m in a.tokens:
 routes=m*8
 if not 0<routes<=4096:raise ValueError('Bounded routes required')
 ids=torch.tensor(pattern(routes,'uniform',19),device='cuda',dtype=dtype).view(m,8)
 def scored():
  light,heavy,counts=base.partition_routes(ids,a.experts,cutoff)
  parts={'light':light,'heavy':heavy}
  return {key:moe_align_block_size(parts[key[0]],key[1],a.experts,ignore_invalid_expert=True) for key in keys},counts
 def candidate():return fused_partition_align(ids,a.experts,cutoff,keys)
 ga,oa=capture(scored);gb,ob=capture(candidate)
 row={'m':m,'routes':routes,'replays':[],'timings':[]}
 for i,name in enumerate(('uniform','hot_rank','hot_expert','threshold','all_remote','uniform')):
  values=pattern(routes,name,31+i);ids.copy_(torch.tensor(values,device='cuda',dtype=dtype).view(m,8));ga.replay();gb.replay();torch.cuda.synchronize();check(*oa,values);check(*ob,values)
  row['replays'].append({'pattern':name,'both_producers_passed':True})
 del ga,gb,oa,ob;torch.cuda.synchronize()
 # Repeat the graph body32 times to amortize host enqueue/event overhead.
 ta,_=capture(scored,32);tb,_=capture(candidate,32)
 for name in ('uniform','hot_rank','all_remote'):
  values=pattern(routes,name,251);ids.copy_(torch.tensor(values,device='cuda',dtype=dtype).view(m,8))
  samples={key:[] for key in ('A1','B1','A2','B2')}
  for order in (list(samples),list(reversed(samples))):
   for label in order:sample(ta if label[0]=='A' else tb)
  for rep in range(a.samples):
   for label in (list(samples) if rep%2==0 else list(reversed(samples))):samples[label].append(sample(ta if label[0]=='A' else tb))
  med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in 'AB'}
  row['timings'].append({'pattern':name,'samples_us':samples,'medians_us':arms,'speedup':arms['A']/arms['B'],'controls':{k:med[k+'1']/med[k+'2'] for k in 'AB'},'noise_pass':all(abs(med[k+'1']/med[k+'2']-1)<=.1 for k in 'AB')})
 del ta,tb;torch.cuda.synchronize();row['passed']=True;report['rows'].append(row);save();print(json.dumps({'m':m,'passed':True,'timings':[{k:v for k,v in t.items() if k!='samples_us'} for t in row['timings']]}),flush=True)
 if (a.out/'STOP_AFTER_CASE').exists():report['status']='stopped_cooperatively';save();sys.exit(0)
report.update(status='completed',passed=True);save();torch.cuda.synchronize()
