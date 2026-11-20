"""Source-only BN narrowing audit and synthetic CTA accounting. No Humming imports."""
import ast,copy,hashlib,json,math,random,statistics
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent.parent;MOE=ROOT/'moe';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=MOE/'humming_vendor/humming/kernel/humming.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HummingKernel');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='check_shape')
ns={'MmaType':SimpleNamespace(WGMMA='wgmma'),'jit_utils':SimpleNamespace(is_power_of_two=lambda x:x>0 and x&(x-1)==0)}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
def check(meta,c):
 s=SimpleNamespace(problem_shape=(0,meta['shape_n'],meta['shape_k']),block_shape=c['block_shape'],warp_shape=c['warp_shape'],mma_type=meta['mma_type'],pad_shape=(0,meta['pad_shape_n'],meta['pad_shape_k']),a_dtype=SimpleNamespace(num_bits=16))
 ns['check_shape'](s)
 assert math.prod(s.block_shape)//math.prod(s.warp_shape)*32<=1024
 assert c['num_stages']>=3
 assert c['use_f16_accum'] is False and c['use_stream_k'] is False

def narrow_columns(meta,config,target=128):
 """Independent tuning prototype; never changes runtime valid_shape_m or packing."""
 if meta['a_dtype']!='bfloat16' or meta['mma_type']!='wgmma':raise ValueError('BF16 WGMMA only')
 c=copy.deepcopy(config)
 if target>c['block_shape'][1]:raise ValueError('Narrowing only')
 c['block_shape']=list(c['block_shape']);c['block_shape'][1]=target
 check(meta,c)
 return c

def main():
 audit=json.loads((MOE/'sm90_wgmma_metadata_audit.json').read_text());legal=[]
 for stage,entry in audit['stages'].items():
  for lo,hi,c in entry['intervals']:
   for target,wn in ((64,32),(64,16),(128,32)):
    proposal=copy.deepcopy(c);proposal['block_shape']=list(proposal['block_shape']);proposal['block_shape'][1]=target;proposal['warp_shape']=list(proposal['warp_shape']);proposal['warp_shape'][1]=wn
    try:check(entry['meta'],proposal);passed=True;reason=''
    except AssertionError as exc:passed=False;reason=str(exc)
    legal.append(dict(stage=stage,interval=[lo,hi],target_n=target,warp_n=wn,passed=passed,reason=reason))
 assert all(not r['passed'] for r in legal if r['target_n']==64)
 assert all(r['passed'] for r in legal if r['target_n']==128)
 rows=[]
 for m in (1,2,4,8,16,33,128,256):
  for pattern in ('uniform','hot_rank','hot_expert','all_remote'):
   for stage,entry in audit['stages'].items():
    old=next(c for lo,hi,c in entry['intervals'] if lo<m*8<=hi);new=narrow_columns(entry['meta'],old);bm=old['block_shape'][0];assert bm==new['block_shape'][0] and old['block_shape'][2]==new['block_shape'][2] and old['warp_shape']==new['warp_shape']
    counts=[]
    for seed in range(256):
     rng=random.Random(seed);hist=[0]*32
     for _ in range(m):
      routes=rng.sample(range(256),8) if pattern=='uniform' else rng.sample(range(32),8) if pattern=='hot_rank' else [0]+rng.sample(range(1,256),7) if pattern=='hot_expert' else rng.sample(range(32,256),8)
      for e in routes:
       if e<32:hist[e]+=1
     rowblocks=sum((n+bm-1)//bm for n in hist);a=rowblocks*entry['meta']['shape_n']//old['block_shape'][1];b=rowblocks*entry['meta']['shape_n']//new['block_shape'][1]
     counts.append((sum(hist),sum(n>0 for n in hist),a,b,min(a,132)/132,min(b,132)/132,a<132 and a>0,b<132 and b>0))
    means=[statistics.mean(x[i] for x in counts) for i in range(8)]
    rows.append(dict(m=m,pattern=pattern,stage=stage,block_m=bm,old_bn=old['block_shape'][1],new_bn=128,mean_local_routes=means[0],mean_active_local_experts=means[1],mean_old_ctas=means[2],mean_new_ctas=means[3],mean_old_first_wave_coverage=means[4],mean_new_first_wave_coverage=means[5],fraction_nonzero_under132_old=means[6],fraction_nonzero_under132_new=means[7]))
 report=dict(status='BN64_REJECTED_BN128_W13_SOURCE_FEASIBLE_UNBENCHMARKED',scope='Exact extracted Python shape guards and synthetic route accounting only; no CUDA/compiler/Humming import',source_hashes={str(p.relative_to(MOE)):sha(p) for p in (source,MOE/'humming_vendor/humming/transform.py',MOE/'humming_vendor/humming/tune/sm90.py',MOE/'sm90_wgmma_metadata_audit.json')},guard_cases=legal,seed_count=256,geometry=dict(local_experts=32,global_experts=256,top_k=8,hidden=4096,intermediate=4096,sms=132),rows=rows,limitations=['CTA counts mean useful output-tile work assignments, not launched grid size. Launcher uses fixed num_ctas_per_sm*SMs; Stream-K-off scheduler distributes tiles round-robin. First-wave useful-work coverage is not measured occupancy or runtime speedup.','BN64 requires changing unsupported BF16 warp geometry; rejected.','BN128 changes w13 only; w2 already uses128 in small-M settings.','Smaller tiles duplicate A input reads and CTA overhead; hot-rank cases already supply enough CTAs.','Same blockM/K, warpM/N/K, dtype, FP32 accumulation, Stream-K-off and pipeline stages retained.','Weight/scale transform APIs consume no block/warp tuning shape; packed ownership and runtime output/cache sizes unchanged.','Hopper compile, numerical validation and cold timing required before promotion.'])
 (OUT/'BN_NARROWING_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'guard_cases':len(legal),'routing_summary_rows':len(rows),'small_uniform':[r for r in rows if r['pattern']=='uniform' and r['m']<=8]}))
if __name__=='__main__':main()
