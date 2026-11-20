from pathlib import Path
import ast,copy,hashlib,json,sys,types
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from w13_h200_contract import approved_job,validate_tensors,require_h200
jobs=json.loads((P/'w13_fused_jobs.json').read_text())['jobs'];checks=[];negatives=0
class Tensor:
 def __init__(self,shape,dtype,ptr):self.shape=tuple(shape);self.dtype='torch.'+dtype;self.device='cuda:0';self.contiguous=True;self.pointer=ptr
 def is_contiguous(self):return self.contiguous
 def numel(self):
  v=1
  for x in self.shape:v*=x
  return v
 def data_ptr(self):return self.pointer
for job in jobs:
 c=job['configuration'];approved_job(job['job_id'],c);m=job['tokens'];bm=c['block_shape'][0];capacity=((m*8+32*(bm-1)+bm-1)//bm)*bm
 x=Tensor((m,4096),'bfloat16',1);out=Tensor((m*8,4096),'bfloat16',2);ids=Tensor((m,8),'int32',3);aligned=[Tensor((capacity,),'int32',4),Tensor((capacity//bm,),'int32',5),Tensor((1,),'int32',6)]
 layer=types.SimpleNamespace(w13_post_scale=Tensor((32,),'bfloat16',7),w13_weight=Tensor((32,8192,512),'int32',8),w13_weight_scale=Tensor((32,8192,256),'float8_e4m3fn',9),locks=Tensor((2048,),'int32',10))
 validate_tensors(layer,x,out,ids,aligned,c);checks.append(job['job_id'])
 for label,mutate in [('output_width',lambda l,x,o,i,a:setattr(o,'shape',(m*8,8192))),('activation_dtype',lambda l,x,o,i,a:setattr(x,'dtype','torch.float16')),('ids_dtype',lambda l,x,o,i,a:setattr(i,'dtype','torch.int64')),('alignment_stride',lambda l,x,o,i,a:setattr(a[0],'contiguous',False)),('weight_device',lambda l,x,o,i,a:setattr(l.w13_weight,'device','cuda:1')),('scale_shape',lambda l,x,o,i,a:setattr(l.w13_post_scale,'shape',(31,))),('packed_weight_size',lambda l,x,o,i,a:setattr(l.w13_weight,'shape',(1,))),('packed_scale_dtype',lambda l,x,o,i,a:setattr(l.w13_weight_scale,'dtype','torch.bfloat16')),('padded_rank',lambda l,x,o,i,a:setattr(a[2],'shape',(1,1))),('alignment_capacity',lambda l,x,o,i,a:setattr(a[0],'shape',(capacity-1,))),('output_alias',lambda l,x,o,i,a:setattr(o,'pointer',x.pointer))]:
  params=copy.deepcopy((layer,x,out,ids,aligned));mutate(*params)
  try:validate_tensors(*params,c)
  except ValueError:negatives+=1
  else:raise AssertionError(label)
 for field,value in [('use_tma_b',True),('use_cp_async',False),('num_experts',4),('use_stream_k',True)]:
  changed=copy.deepcopy(c);changed[field]=value
  try:approved_job(job['job_id'],changed)
  except ValueError:negatives+=1
  else:raise AssertionError(field)
for args in ((8,6,'RTX 3050'),(9,0,'H100'),(10,0,'B200')):
 try:require_h200(*args)
 except RuntimeError:negatives+=1
 else:raise AssertionError(args)
for f in ('w13_fused_gate_h200.py','w13_h200_precompiled.py','w13_h200_contract.py'):ast.parse((P/f).read_text(encoding='utf-8-sig'))
source=(P/'w13_fused_gate_h200.py').read_text(encoding='utf-8-sig');assert source.index('require_h200(prop.')<source.index('from kernel_runtime import bootstrap')
wrapper=(P/'w13_h200_precompiled.py').read_text(encoding='utf-8-sig');assert 'self.kernel_filename=str(cubin);self.load_cubin()' in wrapper and 'KernelRuntime.prepare(' not in wrapper and 'super().prepare(' not in wrapper
root=(P.parent/'coordination/w13_fused_epilogue.py').read_text();assert "assert self.sm_version==86" in root
rtree=ast.parse(root);klass=next(n for n in rtree.body if isinstance(n,ast.ClassDef) and n.name=='FusedW13Kernel');run=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='run');values=next(n.value for n in run.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='values' for t in n.targets));assert len(values.elts)==18
r={'status':'CPU_contract_passed','valid_compiled_configs':checks,'negative_guards':negatives,'explicit_driver_argument_count':len(values.elts),'root_SM86_runtime_guard_unchanged':True,'no_fused_compiler_path':True,'architecture_guard_before_bootstrap':True,'source_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('w13_fused_gate_h200.py','w13_h200_precompiled.py','w13_h200_contract.py')},'limits':'CPU AST and fake metadata tests only; no Torch import, CUDA context, compiler, kernel launch or H200 numerical evidence.'}
(P/'W13_H200_GATE_CPU_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
