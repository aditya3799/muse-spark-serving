"""Bounded exact-source router arithmetic and local two-stream headroom probe."""
from pathlib import Path
import argparse,ast,hashlib,json,os,statistics,sys,time
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokens',type=int,nargs='+',default=[64,512,4096,16384]);p.add_argument('--samples',type=int,default=12);a=p.parse_args()
assert os.environ.get('ROUTER_PROBE_BOUND')=='360';assert a.samples>=12 and a.samples%2==0
assert all(m>8 and m%8==0 for m in a.tokens)
a.out.mkdir(parents=True,exist_ok=True);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'manifest.json').read_text());sources={k:a.workspace/v['relative_workspace_path'] for k,v in manifest['sources'].items()}
for name,path in sources.items():assert sha(path)==manifest['sources'][name]['sha256']
import torch
from torch import nn
import torch.nn.functional as F
torch.set_grad_enabled(False);torch.manual_seed(517)
ns={'torch':torch,'nn':nn,'F':F};tree=ast.parse(sources['model'].read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSpark1xRouter');exec(compile(ast.Module(body=[node],type_ignores=[]),str(sources['model']),'exec'),ns);Router=ns[node.name]
node=next(n for n in ast.parse(sources['projection'].read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='prefill_row_projection');exec(compile(ast.Module(body=[node],type_ignores=[]),str(sources['projection']),'exec'),ns);project=ns[node.name]
class LocalOnly:
 world_size=8;rank_in_group=7
 def all_gather(self,x,dim=0):assert dim==0;return x
report={'status':'running','harness_sha256':sha(Path(__file__)),'manifest_sha256':sha(P/'manifest.json'),'source_hashes':{k:sha(v) for k,v in sources.items()},'argv':sys.argv,'scope':manifest['scope'],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'geometry':{'K':8192,'pooler_N':4096,'global_experts':256,'partitions':8,'tokens':a.tokens},'numeric':[],'timings':[],'routing_available':False,'fixed_gates':{'relative_l2':.003,'max_abs':.03125,'topk_ids':'exact','routing_score_abs':1e-6,'noise_ratio_deviation':.10}}
def save():(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
router=Router(8192,256,torch.float32).cuda();router.weight.copy_(torch.randn_like(router.weight)/90);original_weight=router.weight.clone();router._build_split2()
weight=torch.randn((4096,8192),device='cuda',dtype=torch.bfloat16)/32
flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8);side=torch.cuda.Stream();group=LocalOnly()
route=None
try:
 sys.path.insert(0,str(a.workspace/'sglang/python'))
 from sglang.kernels.ops.moe.muse_spark_routing import FusedMuseSparkRouting
 route=FusedMuseSparkRouting(router.e_score_correction_bias,route_scale=1.0,router_output_multiplier=.17677669529663687,router_score_division_eps=1e-15)
 # Warm real selected fused operation outside graphs; failure means no route proof.
 route(None,router.forward_split2(torch.zeros((64,8192),device='cuda',dtype=torch.bfloat16)),8,True,want_packed=True)
 report['routing_available']=True;report['routing_module']=__import__('inspect').getfile(FusedMuseSparkRouting)
except (ImportError,ModuleNotFoundError) as exc:
 report['routing_import_error']=repr(exc);route=None
print(json.dumps({'milestone':'weights_and_routing_ready','routing_available':route is not None}),flush=True);save()
def check(full,joined,m,pattern,mode):
 diff=joined-full;rel=(diff.norm()/full.norm().clamp_min(1e-20)).item();absolute=diff.abs().max().item()
 row={'m':m,'pattern':pattern,'mode':mode,'relative_l2':rel,'max_abs':absolute,'logits_bitwise':bool(torch.equal(full.view(torch.int32),joined.view(torch.int32))),'passed':bool(torch.isfinite(joined).all()) and rel<=.003 and absolute<=.03125}
 if route is not None:
  ra=route(None,full,8,True,want_packed=True);rb=route(None,joined,8,True,want_packed=True);torch.cuda.synchronize();row.update(topk_ids_exact=bool(torch.equal(ra[1],rb[1])),routing_score_max_abs=(ra[0]-rb[0]).abs().max().item(),packed_exact=bool(torch.equal(ra[2],rb[2])))
  row['passed']=row['passed'] and row['topk_ids_exact'] and row['routing_score_max_abs']<=1e-6 and row['packed_exact']
 report['numeric'].append(row);save()
 if not row['passed']:raise AssertionError(row)
def capture(fn,repeat=1):
 for _ in range(3):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(repeat):out=fn()
 return g,out
for m in a.tokens:
 x=torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)*.05;local=x[7*m//8:]
 # Both calls always use the original globally chosen split2 method, even local M<=8.
 def full_router():return router.forward_split2(x)
 def all_rows():return torch.cat([router.forward_split2(x[r*m//8:(r+1)*m//8]) for r in range(8)],dim=0)
 gf,of=capture(full_router);gs,osh=capture(all_rows)
 for pattern in ('random','zeros','duplicate_weights','near_ties'):
  torch.manual_seed(521)
  router.weight.copy_(original_weight)
  if pattern in ('duplicate_weights','near_ties'):
   router.weight[1::2].copy_(router.weight[::2])
   if pattern=='near_ties':router.weight[1::2,0].add_(1e-7)
  router._build_split2()
  # Packed tensor replacement needs recapture; this is deliberate reload coverage.
  del gf,gs,of,osh;torch.cuda.synchronize();gf,of=capture(full_router);gs,osh=capture(all_rows)
  if pattern=='zeros':x.zero_()
  else:x.copy_(torch.randn_like(x)*.05)
  eager_a=full_router();eager_b=all_rows();check(eager_a,eager_b,m,pattern,'eager');del eager_a,eager_b
  gf.replay();gs.replay();torch.cuda.synchronize();check(of,osh,m,pattern,'changed_input_graph')
 del gf,gs,of,osh;torch.cuda.synchronize();router.weight.copy_(original_weight);router._build_split2();x.copy_(torch.randn_like(x)*.05)
 def pooled():return project(group,x,weight)
 def local_router():return router.forward_split2(local)
 def joined(local_rows):
  main=torch.cuda.current_stream();side.wait_stream(main)
  pool=pooled()
  with torch.cuda.stream(side):logits=router.forward_split2(local if local_rows else x)
  main.wait_stream(side)
  return pool,logits
 funcs={'A':lambda:joined(False),'B':lambda:joined(True),'P':pooled,'R':full_router,'L':local_router}
 # Repeat inside graph only for tiny shape so event intervals exceed host enqueue gaps.
 repeat=8 if m<=64 else 1
 graphs={};outputs={}
 for label,fn in funcs.items():graphs[label],outputs[label]=capture(fn,repeat)
 torch.cuda.synchronize();assert torch.equal(outputs['A'][0],outputs['B'][0])
 assert torch.allclose(outputs['B'][1],outputs['A'][1][7*m//8:],rtol=.003,atol=.03125)
 def sample(g):
  flush.zero_();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();g.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000/repeat
 labels=[k+suffix for suffix in ('1','2') for k in funcs];samples={k:[] for k in labels}
 for order in (labels,labels[::-1]):
  for label in order:sample(graphs[label[0]])
 for rep in range(a.samples):
  for label in (labels if rep%2==0 else labels[::-1]):samples[label].append(sample(graphs[label[0]]))
 med={k:statistics.median(v) for k,v in samples.items()};arms={k:statistics.median(samples[k+'1']+samples[k+'2']) for k in funcs};controls={k:med[k+'1']/med[k+'2'] for k in funcs}
 report['timings'].append({'m':m,'production_row_projection_eligible_size':m>=512,'samples_us':samples,'median_us':arms,'controls':controls,'noise_pass':all(abs(v-1)<=.1 for v in controls.values()),'graph_body_repeats':repeat,'gross_headroom_us':arms['A']-arms['B'],'full_router_above_pooler_us':arms['A']-arms['P'],'local_router_above_pooler_us':arms['B']-arms['P'],'new_logits_gather_output_bytes':m*256*4,'ring_gather_bytes_per_rank':m*256*4*7//8,'labels':{'A':'main local pooler + side full router, joined','B':'main local pooler + side rank-local router, joined','P':'local pooler only','R':'full router only','L':'rank-local router only'},'scope':'No routing operation in timing; no projection all-gather or logits all-gather. A-B is gross local headroom, not serving speedup. Repeated tiny graph bodies warm weights after first replay; 64MiB pressure is not measured residency.'})
 print(json.dumps({'milestone':'shape_complete','m':m,'gross_headroom_us':report['timings'][-1]['gross_headroom_us'],'noise_pass':report['timings'][-1]['noise_pass']}),flush=True)
 del graphs,outputs,x,local;torch.cuda.synchronize();save()
report.update(status='completed',passed=True,timing_noise_failures=sum(not r['noise_pass'] for r in report['timings']));save();print(json.dumps({'status':'completed','numeric_checks':len(report['numeric'])}),flush=True)
