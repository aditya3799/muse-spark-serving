import ast,json,hashlib,types
from pathlib import Path
p=Path(__file__).resolve().parent
out=p/'serving_humming_v1';rel=Path('python/sglang/srt/layers/muse_spark')
src=out/rel/'muse_spark_nvfp4_moe.py';tree=ast.parse(src.read_text(encoding='utf-8-sig'))
c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MuseSparkNvFp4FusedMoEMethod')
init=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
state={};delegated=[]
class Base:
 def __init__(self,config): delegated.append(config)
def fakeimport(name,*args,**kwargs):
 if name=='sglang.srt.layers.moe':return types.SimpleNamespace(get_moe_runner_backend=lambda:types.SimpleNamespace(is_humming=lambda:state['humming']))
 if name=='sglang.srt.utils.common':return types.SimpleNamespace(is_cuda=lambda:state['cuda'],get_device_capability=lambda:state['cap'])
 raise AssertionError(name)
ns={'Base':Base,'__builtins__':dict(vars(__import__('builtins')),__import__=fakeimport)}
cl=ast.ClassDef(name='ExactConstructor',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[init],decorator_list=[])
exec(compile(ast.fix_missing_locations(ast.Module(body=[cl],type_ignores=[])),'<exact-export-constructor>','exec'),ns)
rows=[]
for humming,cuda,cap,expected in [(True,True,(8,6),'pass'),(True,True,(9,0),'pass'),(True,False,(9,0),'reject'),(True,True,(7,5),'reject'),(True,True,(10,0),'reject'),(False,True,(9,0),'delegate'),(False,False,(0,0),'delegate')]:
 state.update(humming=humming,cuda=cuda,cap=cap);cfg=object();before=len(delegated)
 try:
  obj=ns['ExactConstructor'](cfg)
  result='delegate' if len(delegated)>before else 'pass'
  if result=='pass':
   assert obj.quant_config is cfg and obj._lora_marlin is False
   assert obj.enable_flashinfer_trtllm_moe is False and obj.use_flashinfer_trtllm_weight_layout is False
   assert obj._cache_permute_indices=={}
 except ValueError:result='reject'
 assert result==expected,(humming,cuda,cap,result,expected)
 rows.append(dict(humming=humming,cuda=cuda,cap=cap,result=result))
report={'scope':'Exact exported constructor with injected backend/device interfaces; no GPU or arithmetic stubs','passed':True,'rows':rows}
(p/'constructor_test.json').write_text(json.dumps(report,indent=2))
manifest=json.loads((out/'manifest.json').read_text())
for file in (out/rel).glob('*.py'):
 text=file.read_text(encoding='utf-8-sig');ast.parse(text);file.write_bytes(text.replace('\r\n','\n').encode('utf-8'))
 manifest['files'][str(file.relative_to(out)).replace('\\','/')]=hashlib.sha256(file.read_bytes()).hexdigest()
manifest['constructor_test']='../constructor_test.json'
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(report))
