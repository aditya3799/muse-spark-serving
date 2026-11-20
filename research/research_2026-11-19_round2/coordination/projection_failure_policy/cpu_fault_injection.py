"""CPU fault injection into actual amended AST; mocked CPU collectives, no CUDA/NCCL."""
import ast,json,pathlib,threading,types,hashlib,concurrent.futures
HERE=pathlib.Path(__file__).resolve().parent;SRC=HERE/'sources/python/sglang/srt/speculative'
ns={};exec((SRC/'dflash_lt_status.py').read_text(),ns);checked=ns['checked_vendor_status'];VendorError=ns['ProjectionVendorError']
class OOM(Exception):pass
class Scalar:
 def __init__(self,v):self.v=int(v)
 def __int__(self):return self.v
 def __bool__(self):return bool(self.v)
class Tensor:
 def __init__(self,shape=(512,1024)):
  self.shape=shape;self.dtype='bf16';self.device='cpu';self._version=0
 def data_ptr(self):return 256
 def stride(self):return (self.shape[-1],1)
 def numel(self):return 8
class Plan:
 def __init__(self):self.closed=False;self.scratch=Tensor();self.device='cpu';self.stream=17;self.workspace=Tensor()
 def close(self):self.closed=True
class Collective:
 def __init__(self,w):self.w=w;self.barrier=threading.Barrier(w,timeout=10);self.lock=threading.Lock();self.seq=[0]*w;self.data={};self.calls=[]
 def exchange(self,value,group,kind):
  r=group.rank;i=self.seq[r];self.seq[r]+=1
  with self.lock:self.data.setdefault(i,{})[r]=(kind,value);self.calls.append((r,i,kind))
  self.barrier.wait()
  values=[self.data[i][q][1] for q in range(self.w)]
  assert all(self.data[i][q][0]==kind for q in range(self.w))
  self.barrier.wait();return values
 def all_gather_object(self,out,value,group):out[:]=self.exchange(value,group,'gather')
 def all_reduce(self,tensor,op,group):tensor.v=min(self.exchange(int(tensor),group,'min'))
class Entry:
 def __init__(self,*args):self.args=args;self.metadata=args[-1]
def extract(path,cls,method,env):
 tree=ast.parse(path.read_text());c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls);f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
 module=ast.Module(body=[f],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),str(path),'exec'),env);return env[method]

def run(kind,world,fault):
 comm=Collective(world);plans=[];lock=threading.Lock();results=[None]*world
 fake_torch=types.SimpleNamespace(OutOfMemoryError=OOM,int32='i32',uint8='u8',tensor=lambda x,**kw:Scalar(x),empty=lambda *a,**kw:Tensor(),cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,current_stream=lambda:types.SimpleNamespace(cuda_stream=17)))
 env=dict(torch=fake_torch,dist=types.SimpleNamespace(all_gather_object=comm.all_gather_object,all_reduce=comm.all_reduce,ReduceOp=types.SimpleNamespace(MIN='min')),_stream=lambda x:17,runtime_state=lambda:(('cublaslt',8),True,False,False),weight_version=lambda x:0,Entry=Entry,discover=None)
 method=extract(SRC/('dflash_projection_row.py' if kind=='row' else 'dflash_projection_n/adapter.py'),'RowProjectionRouter' if kind=='row' else 'OutputNRouter','register',env)
 def worker(rank):
  cpu=types.SimpleNamespace(rank=rank);group=types.SimpleNamespace(world_size=world,rank_in_group=rank,cpu_group=cpu)
  w=Tensor((8192,1024));original=types.SimpleNamespace(weight=w)
  obj=types.SimpleNamespace(original=original,group=group,registration_index=0,registration_ready=True,registration_manifest=[(512,1024)],plans={},published=set(),workspaces={},closed=False,entries={},manifest=[(512,1024)],mode='qualified',stream_workspaces={})
  def discover(examples,size):
   if rank==world-1:
    if fault=='unsupported':return None,dict(reason='unsupported')
    if fault=='oom':raise OOM('injected recoverable allocation')
    if fault=='programming':raise RuntimeError('injected programming failure')
    if isinstance(fault,int):checked(fault,'injected Lt call')
   plan=Plan()
   with lock:plans.append(plan)
   return plan,dict(chosen=True)
  try:
   result=method(obj,[Tensor(),Tensor()],discover_fn=discover)
   results[rank]=dict(aborted=False,enabled=result.get('accepted',result.get('globally_enabled',False)))
  except RuntimeError as e:results[rank]=dict(aborted=True,error=str(e))
 with concurrent.futures.ThreadPoolExecutor(max_workers=world) as pool:list(pool.map(worker,range(world)))
 fatal=fault=='programming' or isinstance(fault,int)
 assert all(r['aborted']==fatal for r in results),(kind,world,fault,results)
 if fatal:
  assert all(p.closed for p in plans)
  assert not any(k=='min' for _,_,k in comm.calls)
 else:
  assert all(r['enabled']==(fault=='none') for r in results)
  if fault!='none':assert all(p.closed for p in plans)
 return dict(kind=kind,world=world,fault=fault,results=results,collective_rounds=max(comm.seq),all_created_plans_closed=all(p.closed for p in plans) if fault!='none' else None)

status_cases=[]
for status in (0,15,1,3,7,8,11,13,14,16,-1,99):
 try:value=checked(status,'unit');outcome='returned'
 except VendorError as e:assert e.status==status;outcome='abort'
 assert (outcome=='returned')==(status in(0,15));status_cases.append(dict(status=status,outcome=outcome))
# Exercise both real invoke methods with an injected C ABI return, not only helper.
invoke_cases=[]
for kind in ('row','column'):
 path=SRC/('dflash_projection_row.py' if kind=='row' else 'dflash_projection_n/runtime_projection.py')
 env=dict(checked_vendor_status=checked,_stream=lambda x:17,torch=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda:types.SimpleNamespace(cuda_stream=17))))
 method=extract(path,'RowPlan' if kind=='row' else 'VendorPlan','invoke',env)
 for status in (0,15,7,13,14,99):
  lib=types.SimpleNamespace(row_plan_call=lambda *args,s=status:s,projection_call=lambda *args,s=status:s);obj=types.SimpleNamespace(lib=lib,handle=123,scratch=Tensor())
  try:
   result=method(obj,0,False,Tensor(),Tensor(),Tensor()) if kind=='row' else method(obj,0,False,Tensor(),Tensor(),Tensor(),Tensor())
   assert result==status;outcome='returned'
  except VendorError:outcome='abort'
  assert (outcome=='returned')==(status in (0,15));invoke_cases.append(dict(kind=kind,status=status,outcome=outcome))
cases=[run(kind,w,fault) for kind in ('row','column') for w in (2,4) for fault in ('none','unsupported','oom','programming',1,3,7,8,13,14,99)]
report=dict(scope=__doc__,complete=True,all_passed=True,status_cases=status_cases,actual_invoke_cases=invoke_cases,registration_cases=cases,rank_cases=sum(x['world'] for x in cases),patch_sha256=hashlib.sha256((HERE/'incremental.diff').read_bytes()).hexdigest(),source_hashes={str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob('*.py')})
(HERE/'cpu_fault_injection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=True,status_cases=len(status_cases),invoke_cases=len(invoke_cases),registration_scenarios=len(cases),rank_cases=report['rank_cases'])))
