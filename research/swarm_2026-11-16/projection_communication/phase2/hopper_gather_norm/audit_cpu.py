from pathlib import Path
import ast,copy,hashlib,json,re,types,typing
ROOT=Path(r'D:\OpenKernelChallenge\serving');OUT=Path(__file__).resolve().parent;CAND=OUT/'candidate_sources';BASE=ROOT/'sglang'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
np='python/sglang/kernels/ops/layernorm/muse_spark_norm_residual_jit.py'
constants={'_VEC','_THREAD_CHOICES','_CLUSTER_SIZES','_SMALL_M','_SMALL_M_THREAD_CHOICES'}
functions={'_launch_config','cluster_size_for_num_tokens','_cluster_num_threads','_cluster_size_for_shape','norm_residual_launch_config','_cluster_launch_config'}
def policy(path):
 ns={'Optional':typing.Optional};nodes=[]
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets):nodes.append(n)
  if isinstance(n,ast.FunctionDef) and n.name in functions:nodes.append(n)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns);return ns
old=policy(BASE/np);new=policy(CAND/np);policy_cases=0
for h in (1024,2048,4096,8192,16384):
 for m in range(514):
  assert new['norm_residual_launch_config'](h,m)==old['norm_residual_launch_config'](h,m)
  assert new['norm_residual_launch_config'](h,m,use_clusters=False)==(1,*old['_launch_config'](h,m))
  policy_cases+=1
stock=(BASE/'python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh').read_text()
gather=(BASE/'python/sglang/kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh').read_text()
stock=stock[stock.index('muse_spark_norm_residual_kernel('):stock.index('// Multi-CTA path')]
gather=gather[gather.index('muse_spark_all_gather_norm_residual_row_kernel('):gather.index('using host::distributed::CommunicatorRef;')]
def norm(text):
 text=text.replace('params.','norm.')
 replacements={
  'v = muse_ag_norm::mul_pair(v, post_w[p][q]);':'v.x *= post_w[p][q].x; v.y *= post_w[p][q].y;',
  'normed_b = muse_ag_norm::mul_pair(normed_b, {post_rstd, post_rstd});':'normed_b.x *= post_rstd; normed_b.y *= post_rstd;',
  'normed_b = muse_ag_norm::mul_pair(normed_b, post_w[p][q]);':'normed_b.x *= post_w[p][q].x; normed_b.y *= post_w[p][q].y;',
  'v = muse_ag_norm::mul_pair(v, {pre_rstd, pre_rstd});':'v.x *= pre_rstd; v.y *= pre_rstd;',
  'v = muse_ag_norm::mul_pair(v, wf);':'v.x *= wf.x; v.y *= wf.y;',
  'F32Vec router_v[kFParts];':'',
  'if constexpr (kHasRouter) {\n        router_v[q / kFPairs][q % kFPairs] = v;\n      }':''}
 for a,b in replacements.items():text=text.replace(a,b)
 return re.sub(r'\s+','',text)
segments=[]
for label,start,end in [('post_norm','auto v = cast<fp32x2_t>(branch_v[q]);','const float post_rstd ='),('gated_residual_and_pre_reduction','fp32x2_t gate_a =','const float pre_rstd ='),('pre_norm_output','const float pre_rstd =','out_v.store(out, idx);')]:
 def section(t):
  i=t.index(start);j=t.index(end,i)
  if end.startswith('const float'):j=t.index(';',j)+1
  else:j+=len(end)
  return norm(t[i:j])
 a,b=section(stock),section(gather)
 assert a==b,(label,a[:160],b[:160])
 segments.append({'name':label,'normalized_operations_sha256':hashlib.sha256(a.encode()).hexdigest(),'equal':True})
# Exhaustive byte-vector address ownership, without simulating remote memory ordering.
geometry=[]
for h in (4096,8192):
 for world in (2,4,8):
  for m in (1,8,9,16,32,64,128,191,192,256,511):
   for counters in (2,16,132):
    _,threads,_=new['norm_residual_launch_config'](h,m,use_clusters=False)
    grid=min(m,counters);local=h//world;vectors=m*h//8
    seen=bytearray(vectors)
    for block in range(grid):
     for row in range(block,m,grid):
      for thread in range(threads):
       for pack in range(h//(threads*8)):
        col=(thread+pack*threads)*8
        physical=((col//local)*m+row)*(local//8)+(col%local)//8
        assert seen[physical]==0;seen[physical]=1
    assert all(seen)
    # Producer loop is a strided partition of each rank's contiguous vectors.
    produced=bytearray(m*local//8)
    for worker in range(grid*threads):
     for vector in range(worker,len(produced),grid*threads):
      assert produced[vector]==0;produced[vector]=1
    assert all(produced)
    updated=list(range(grid))+list(range(grid,counters))
    assert sorted(updated)==list(range(counters)) and len(set(updated))==counters
    geometry.append({'hidden':h,'world':world,'rows':m,'counters':counters,'threads':threads,'grid':grid,'vectors':vectors,'coverage_once':True,'all_counters_advance_once':True})
cp='python/sglang/kernels/ops/communication/muse_spark_all_gather_norm_residual.py'
class CA:pass
arch=[9]
torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda:(arch[0],0)))
ns={'torch':torch,'CustomAllReduceV2':CA,'Optional':typing.Optional,'norm_residual_launch_config':new['norm_residual_launch_config'],'_VEC':8}
nodes=[]
for n in ast.parse((CAND/cp).read_text()).body:
 if isinstance(n,ast.FunctionDef) and n.name in ('_supports_clusters','all_gather_norm_residual_supported'):
  n=copy.deepcopy(n);n.decorator_list=[];n.body=[x for x in n.body if not isinstance(x,(ast.Import,ast.ImportFrom))];nodes.append(n)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CAND/cp),'exec'),ns)
support=ns['all_gather_norm_residual_supported'];guards=0
for major in (8,9,10):
 arch[0]=major
 for m in (0,1,8,64,128,256,511):
  for counters in (2,132):
   comm=CA();comm.disabled=False;comm.has_multicast=True;comm.world_size=8;comm.max_push_size=512*8192*2;comm.config=types.SimpleNamespace(num_push_blocks=counters)
   if major==8 or m==0:expected=False
   else:
    c,_,_=new['norm_residual_launch_config'](8192,m,use_clusters=major>=10);expected=c==1 or m*c<=counters
   assert support(comm,m,8192)==expected;guards+=1
report={'status':'CPU_POLICY_ARITHMETIC_AND_LAYOUT_PASS','launch_policy_cases':policy_cases,'support_cases':guards,'source_arithmetic_segments':segments,'geometry_cases':len(geometry),'geometry':geometry,'source_hashes':{str(p):sha(p) for p in (CAND/np,CAND/cp,BASE/'python/sglang/kernels/jit/csrc/muse_spark/norm_residual.cuh',BASE/'python/sglang/kernels/jit/csrc/muse_spark/all_gather_norm_residual.cuh')},'precision_scope':'Same per-lane FP32 sums/FMA order, CTA reduction functions, BF16 rounding and optional weights/gates after transport. Pair multiplication syntax expanded component-wise for comparison; router-disabled branch matches FFN use. Source equivalence is not compiled SASS or runtime proof.','protocol_scope':'Exact producer/consumer vector mapping and every counter covered once; unchanged zero-marker storage/loads. No remote visibility, lifetime, concurrent residency or multiGPU synchronization proof.','remaining_gates':['OfflineSM90 exact C1 instantiation','Actual H200 multicast/counter/graph replay test','Numerical signed-zero and optional gate/weight runtime cases','Scored-baseline relative latency and whole-model capture memory']}
(OUT/'CPU_AUDIT.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps({k:report[k] for k in ('status','launch_policy_cases','support_cases','geometry_cases')}))