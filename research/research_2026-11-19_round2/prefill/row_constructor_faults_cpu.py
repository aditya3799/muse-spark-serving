"""Execute actual constructor/discover AST with deterministic allocation faults."""
import ast,json,types,hashlib,os,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1];O=H/'row_constructor_cleanup_unapplied'
path=O/'sources/python/sglang/srt/speculative/dflash_projection_row.py'
source=path.read_text();tree=ast.parse(source)
chosen=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RowPlan' or isinstance(n,ast.FunctionDef) and n.name=='discover']
module=ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[]))
class OOM(RuntimeError):pass
class Tensor:
 def __init__(self,shape=(512,1024)):self.shape=shape;self.device='fake';self.dtype='bf16'
 def data_ptr(self):return 16
 def numel(self):return self.shape[0]
 def view(self,*args):return self
 def __getitem__(self,key):return Tensor((64,1024))
records=[]
for name,fault_at,error_cls,choices in [('constructor_oom',1,OOM,1),('constructor_unexpected',1,ValueError,1),('later_discovery_oom',2,OOM,1),('clean_no_plan',None,None,0),('accepted_plan',None,None,1)]:
 calls=[];allocation=[0];injected=error_cls(name) if error_cls else None
 class Lib:
  def make_row_plan(self,*args):calls.append('create');return 123
  def row_plan_destroy(self,handle):assert handle==123;calls.append('destroy')
  def row_plan_count(self,handle):return choices
  def row_plan_attr(self,*args):return 0
  def row_plan_call(self,*args):return 0
 lib=Lib()
 def empty(*args,**kwargs):
  allocation[0]+=1
  if allocation[0]==fault_at:raise injected
  shape=args[0] if isinstance(args[0],tuple) else (args[0],)
  return Tensor(shape)
 torch=types.SimpleNamespace(no_grad=lambda:lambda fn:fn,OutOfMemoryError=OOM,empty=empty,equal=lambda a,b:True,
  uint8='u8',int16='i16',__version__='fake',version=types.SimpleNamespace(cuda='fake'),
  backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_bf16_reduced_precision_reduction=True))),
  cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False,get_device_properties=lambda d:types.SimpleNamespace(name='fake',major=8,minor=6,multi_processor_count=20)))
 env=dict(checked_vendor_status=lambda value,operation:value,torch=torch,eligible=lambda *args:True,_library=lambda:lib,_workspace=lambda:('cublaslt',64),_identity=lambda w:('weight',),_stream=lambda x:7,_policy=lambda:('default',),F=types.SimpleNamespace(linear=lambda x,w:Tensor((x.shape[0],w.shape[0]))))
 exec(compile(module,str(path),'exec'),env)
 caught=None;result=None
 try:result=env['discover']([(Tensor(),Tensor((1024,1024)))],8)
 except Exception as error:caught=error
 if injected is not None:
  assert caught is injected,(name,caught)
  assert calls==['create','destroy'],(name,calls)
 else:
  assert caught is None,(name,repr(caught))
  if choices:
   assert calls==['create']
   result[0].close();result[0].close()
  assert calls==['create','destroy'],(name,calls)
 records.append(dict(case=name,allocation_calls=allocation[0],handle_events=calls,
  same_exception_rethrown=caught is injected if injected else None,exactly_once_destroy=True))
report=dict(passed=True,scope=__doc__,actual_source_sha256=hashlib.sha256(source.encode()).hexdigest(),records=records)
(O/'cpu_constructor_faults.json').write_text(json.dumps(report,indent=2))
# Private index only: no working tree or user index writes.
env=os.environ.copy();env['GIT_INDEX_FILE']=str(O/'validation.index')
patches=[R/'research_2026-11-19/prefill/integration_joint/candidate.diff',
 H.parent/'coordination/projection_failure_policy/incremental.diff',H/'early_projection_unapplied/incremental.diff',H/'row_no_grad_unapplied/incremental.diff',O/'incremental.diff']
commands=[['read-tree','297e54fd105e974cdf401e5b4d20253a22c16f90']]+[['apply','--cached',str(p)] for p in patches]+[['diff','--cached','--check']]
steps=[]
for command in commands:
 result=subprocess.run(['git','-C',str(R/'sglang')]+command,env=env,text=True,capture_output=True)
 steps.append(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
 assert result.returncode==0,steps[-1]
(O/'composition_validation.json').write_text(json.dumps(dict(passed=True,scope='Private index only; five patches compose on pinned source, user index/worktree untouched.',steps=steps),indent=2))
print(json.dumps(dict(passed=True,constructor_cases=len(records),composition_steps=len(steps)),indent=2))


