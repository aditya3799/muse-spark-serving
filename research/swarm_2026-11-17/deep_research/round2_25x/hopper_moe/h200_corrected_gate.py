"""H200-only finite numerical gate of hashed corrected cubins. Never submits or times."""
from pathlib import Path
import argparse,hashlib,json,sys
from h200_corrected_contract import require_h200,load_manifest,local_routes,aligned_routes,validate_tensors
P=Path(__file__).resolve().parent;V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor'

def main():
 pa=argparse.ArgumentParser(description=__doc__);pa.add_argument('--out',type=Path,required=True);pa.add_argument('--family',choices=['virtual','wait1','wm48','all'],default='virtual');pa.add_argument('--jobs',nargs='+');pa.add_argument('--seeds',type=int,default=3);args=pa.parse_args()
 if args.out.exists() or not 1<=args.seeds<=10:raise ValueError('Fresh output and1..10seeds required')
 manifest=load_manifest();pairs=[p for p in manifest['pairs'] if (args.family=='all' or p['family']==args.family) and (not args.jobs or p['job_id'] in args.jobs)]
 if not pairs or args.jobs and set(args.jobs)!={p['job_id'] for p in pairs}:raise ValueError('Unknown selection')
 import torch
 props=torch.cuda.get_device_properties();require_h200(props.major,props.minor,props.name)
 sys.path.insert(0,str(V))
 from humming import ops,dtypes
 from humming.config import WeightScaleType,WeightScale2Type
 from humming.layer import HummingMethod
 from humming.schema import HummingInputSchema,HummingWeightSchema
 args.out.mkdir(parents=True);report={'status':'running','scope':'Finite direct indexed GEMM bitwise/sentinel/changed-graph gate. No timing or serving claim. No Humming GEMM JIT compilation; registers exact precompiled cubins. Weight repacking/launcher support may compile standard helpers.','gpu':props.name,'manifest_sha256':hashlib.sha256((P/'H200_CORRECTED_GATE_MANIFEST.json').read_bytes()).hexdigest(),'NaN_contract':'NaN payload/invalid-operation sign equivalence is not claimed; fixtures require finite outputs.','rows':[]}
 def save():
  t=args.out/'results.tmp';t.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');t.replace(args.out/'results.json')
 def exact(a,b):return bool(torch.equal(a.contiguous().view(torch.int16),b.contiguous().view(torch.int16)))
 save()
 for pair in pairs:
  c=pair['candidate']['configuration'];m=pair['tokens'];e,n,k=c['num_experts'],c['shape_n'],c['shape_k'];slots=m*8;topk=8 if pair['stage']=='w13' else 1
  torch.manual_seed(20261117);layer=torch.nn.Module();packed=torch.randint(0,256,(e,n,k//2),device='cuda',dtype=torch.uint8);scales=torch.pow(2.,torch.randint(-8,4,(e,n,k//16),device='cuda')).to(torch.float8_e4m3fn)
  layer.register_parameter('weight',torch.nn.Parameter(packed.view(torch.int32),requires_grad=False));layer.register_parameter('weight_scale',torch.nn.Parameter(scales,requires_grad=False));del packed,scales
  schema=HummingWeightSchema(b_dtype=dtypes.float4e2m1,bs_dtype=dtypes.float8e4m3,weight_scale_group_size=16,weight_scale_type=WeightScaleType.GROUP,weight_scale_2_type=WeightScale2Type.NONE)
  HummingMethod.prepare_layer_meta(layer,shape_n=n,shape_k=k,weight_schema=schema,input_schema=HummingInputSchema(a_dtype=dtypes.bfloat16),num_experts=e,torch_dtype=torch.bfloat16);HummingMethod.transform_humming_layer(layer,already_padded=True)
  meta=json.loads(layer.humming_metas[''].to_str())
  for field in ('shape_n','shape_k','num_experts','a_dtype','b_dtype','bs_dtype','c_dtype','weight_scale_group_size','weight_scale_type','weight_scale_2_type','use_packed_k_layout'):assert meta[field]==c[field],field
  x=torch.zeros((slots//topk,k),device='cuda',dtype=torch.bfloat16);outputs=[torch.full((slots,n),37.,device='cuda',dtype=torch.bfloat16) for _ in range(2)];locks=torch.zeros(max(2048,props.multi_processor_count*8),device='cuda',dtype=torch.int32);tables=[];routes=[];configs=[]
  for side in ('baseline','candidate'):
   item=pair[side];cfg=item['configuration'];configs.append(cfg);kid,kname=ops.register_kernel(str(P/item['cubin_path']));tables.append(torch.tensor([0,1<<30,kid,0],dtype=torch.int64));arr=aligned_routes(local_routes(m,e,'uniform',0),e,cfg['block_shape'][0]);routes.append([torch.tensor(v if i<2 else [v],device='cuda',dtype=torch.int32) for i,v in enumerate(arr)])
  def call(i):
   si,ei,np=routes[i];validate_tensors(x,layer.weight,layer.weight_scale,outputs[i],si,ei,np,configs[i],slots,topk)
   ops.launch_kernel(configs=tables[i],inputs=x,weight=layer.weight,weight_scale=layer.weight_scale,outputs=outputs[i],sorted_ids=si,expert_ids=ei,num_tokens_padded=np,locks=locks,top_k=topk,valid_shape_m=slots)
  graphs=[]
  for i in range(2):
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):call(i)
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):call(i)
   graphs.append(graph)
  row={'job':pair['job_id'],'family':pair['family'],'baseline_cubin_sha256':pair['baseline']['cubin_sha256'],'candidate_cubin_sha256':pair['candidate']['cubin_sha256'],'cases':[]};report['rows'].append(row);save()
  for seed in range(args.seeds):
   for pattern in ('uniform','one_hot','eight_hot','mixed','all_remote'):
    ids=local_routes(m,e,pattern,seed);active=torch.tensor([v>=0 for v in ids],device='cuda',dtype=torch.bool)
    for i,cfg in enumerate(configs):
     arr=aligned_routes(ids,e,cfg['block_shape'][0])
     for j,v in enumerate(arr):routes[i][j].copy_(torch.tensor(v if j<2 else [v],device='cuda',dtype=torch.int32))
    for profile in ('random','cancellation','subnormal','signed_zero'):
     torch.manual_seed(310+seed)
     if profile=='random':x.copy_(torch.randn_like(x)/8)
     elif profile=='cancellation':x.copy_(torch.arange(k,device='cuda').remainder(2).mul(2).sub(1).to(torch.bfloat16).expand_as(x));x[:,::64]*=32
     elif profile=='subnormal':x.copy_(torch.arange(k,device='cuda').remainder(2).mul(2).sub(1).float().mul(2.**-130).to(torch.bfloat16).expand_as(x))
     else:x.zero_();x[:,::2]=-0.
     for out in outputs:out.fill_(37.)
     call(0);call(1);torch.cuda.synchronize();direct=[out.clone() for out in outputs]
     checks={'finite':all(bool(torch.isfinite(out[active]).all()) for out in direct),'bitwise':exact(direct[0][active],direct[1][active]),'inactive':all(bool((out[~active]==37.).all()) for out in direct)}
     for out in outputs:out.fill_(37.)
     graphs[0].replay();graphs[1].replay();torch.cuda.synchronize();checks['changed_graph']=all(exact(outputs[i],direct[i]) for i in range(2));row['cases'].append({'seed':seed,'pattern':pattern,'profile':profile,'active_routes':sum(v>=0 for v in ids),'checks':checks});save()
     if not all(checks.values()):raise AssertionError(row['cases'][-1])
  del graphs,outputs,direct,x,layer,routes,tables,locks;torch.cuda.synchronize()
 load_manifest();report['status']='finite_component_gate_passed';save();print(json.dumps({'status':report['status'],'pairs':len(report['rows'])}))
if __name__=='__main__':main()
