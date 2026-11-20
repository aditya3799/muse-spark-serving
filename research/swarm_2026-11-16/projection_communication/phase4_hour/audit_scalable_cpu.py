import ast,collections,copy,hashlib,json,random
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent;ROOT=P.parents[2];SWARM=P.parents[1]
stock=SWARM/'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
s=stock.read_text();tree=ast.parse(s)
ns={'deepcopy':copy.deepcopy};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'exact_scored_policy','exec'),ns)
audit=SWARM/'moe/sm90_wgmma_metadata_audit.json';d=json.loads(audit.read_text());layer=NS(w13_post_scale=NS(numel=lambda:32),fp32_configs={k:v['intervals'] for k,v in d['stages'].items()})
rows=[]
for m in (512,513,753,1024,3012,4096,8192,16384):
 plan=ns['choose_partition_configs'](layer,m*8,256)
 r={'m':m,'routes':m*8,'scored_eligible':m<=512 and plan is not None,'scalable_eligible':m<=16384 and plan is not None,'plan':plan,'numerical_risks':[]}
 if plan:
  for stage in ('w13','w2'):
   a=plan['heavy'][stage];b=plan['light'][stage]
   assert all(c['use_f16_accum'] is False and c['use_stream_k'] is False for c in (a,b))
   changes={key:{'heavy':a.get(key),'light':b.get(key)} for key in ('block_shape','warp_shape','num_stages','use_stream_k','use_f16_accum') if a.get(key)!=b.get(key)}
   r['numerical_risks'].append({'stage':stage,'config_changes':changes,'block_K_changes':a['block_shape'][2]!=b['block_shape'][2],'warp_K_changes':a['warp_shape'][2]!=b['warp_shape'][2],'interpretation':'Full K still reduced; changed BK/warpK may change accumulation order. FP32 flags alone are insufficient numerical proof.'})
 rows.append(r)
checks=[];rng=random.Random(417)
for routes in (1,4096,4097,6024,24096,131072):
 for experts in (1,32,256):
  for pattern in ('all_remote','hot','balanced','invalid_mixture'):
   if pattern=='all_remote':ids=[-1]*routes
   elif pattern=='hot':ids=[0]*routes
   elif pattern=='balanced':ids=[i%experts for i in range(routes)]
   else:ids=[(-2,-1,0,experts-1,experts,2**35)[i%6] for i in range(routes)]
   reference=collections.Counter(x for x in ids if 0<=x<experts)
   partials=[]
   for start in range(0,routes,2048):partials.append(collections.Counter(x for x in ids[start:start+2048] if 0<=x<experts))
   counts=[sum(part[e] for part in partials) for e in range(experts)]
   assert counts==[reference[e] for e in range(experts)] and sum(counts)==sum(reference.values())
   for cutoff in (8,40,128):
    light=[x if 0<=x<experts and counts[x]<=cutoff else -1 for x in ids]
    heavy=[x if 0<=x<experts and counts[x]>cutoff else -1 for x in ids]
    assert all((a>=0)+(b>=0)==int(0<=x<experts) and (a if a>=0 else b)==(x if 0<=x<experts else -1) for x,a,b in zip(ids,light,heavy))
   checks.append({'routes':routes,'experts':experts,'pattern':pattern,'all_3_cutoffs_pass':True})
# Dependency accounting and no host route-value reads.
ct=ast.parse((P/'scalable_partition.py').read_text());forbidden=[n.func.attr for n in ast.walk(ct) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('item','cpu','tolist','numpy')];assert not forbidden
out={'status':'PASS','ownership_cases':len(checks),'cutoffs_per_case':3,'checks':checks,'SM90_policy_cases':rows,'scope':'Integer partition/index/ownership proof and exact pinned policy source; CPU model does not execute Triton or validate reduction/GEMM numerics.','source_sha256':{'candidate':hashlib.sha256((P/'scalable_partition.py').read_bytes()).hexdigest(),'scored_policy':hashlib.sha256(stock.read_bytes()).hexdigest(),'SM90_metadata':hashlib.sha256(audit.read_bytes()).hexdigest()},'no_host_route_reads':True}
(P/'SCALABLE_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':'PASS','cases':len(checks),'SM90_plans':[(r['m'],r['scalable_eligible']) for r in rows]}))
