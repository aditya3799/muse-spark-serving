"""Native CPU metadata guard audit; no Torch import or device execution."""
import ast,copy,json
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;source=P/'scale_activation_export/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_scale_activation.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_supports_scaled_activation')
state={'compiling':False,'sm':(8,6)}
torch=SimpleNamespace(bfloat16='bf16',int32='i32',int64='i64',version=SimpleNamespace(hip=None),compiler=SimpleNamespace(is_compiling=lambda:state['compiling']),cuda=SimpleNamespace(get_device_capability=lambda _:state['sm']))
ns={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
class Tensor:
    def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype;self.ndim=len(shape);self.device='cuda:0';self.is_cuda=True;self.contiguous=True
    def is_contiguous(self):return self.contiguous
    def numel(self):
        result=1
        for n in self.shape:result*=n
        return result
base=[Tensor((8,8192),'bf16'),Tensor((8,4096),'bf16'),Tensor((8,),'i32'),Tensor((32,),'bf16')]
cases=[]
def check(label,expected,edit=None):
    args=copy.deepcopy(base);state.update(compiling=False,sm=(8,6));torch.version.hip=None
    if edit:edit(args)
    actual=ns['_supports_scaled_activation'](*args);cases.append({'label':label,'expected':expected,'actual':actual});assert actual==expected
check('bf16_sm86',True);check('int64',True,lambda a:setattr(a[2],'dtype','i64'))
check('sm90',True,lambda a:state.update(sm=(9,0)))
check('cpu',False,lambda a:setattr(a[0],'is_cuda',False));check('sm75',False,lambda a:state.update(sm=(7,5)))
check('hip',False,lambda a:setattr(torch.version,'hip','x'));check('compile',False,lambda a:state.update(compiling=True))
check('input_fp16',False,lambda a:setattr(a[0],'dtype','fp16'));check('scale_fp32',False,lambda a:setattr(a[3],'dtype','fp32'))
check('noncontiguous',False,lambda a:setattr(a[0],'contiguous',False));check('other_device',False,lambda a:setattr(a[1],'device','cuda:1'))
check('id_float',False,lambda a:setattr(a[2],'dtype','fp32'));check('id_count',False,lambda a:setattr(a[2],'shape',(7,)))
check('empty_scale',False,lambda a:setattr(a[3],'shape',(0,)));check('wrong_width',False,lambda a:setattr(a[1],'shape',(8,4095)))
report={'status':'PASS_NATIVE_METADATA_GUARDS','cases':cases,'scope':'Extracted exact exported metadata predicate with dummy metadata objects; not CUDA numerical proof'}
(P/'scale_activation_export/guard_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(cases)}))