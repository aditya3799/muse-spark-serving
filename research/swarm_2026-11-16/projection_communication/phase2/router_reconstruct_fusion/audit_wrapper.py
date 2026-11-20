"""CPU spies execute exact candidate wrapper; no Torch/CUDA import or compile."""
from pathlib import Path
import ast,hashlib,json,types
P=Path(__file__).resolve().parent
source=P/'candidate_sources/python/sglang/kernels/ops/moe/muse_spark_routing_jit.py'
tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('fused_muse_spark_routing_jit','_experts_per_lane','_warps_per_block')]
events=[]
class Tensor:
    device='cuda:fake'
    def __init__(self,shape,dtype='float32',contiguous=True):self.shape=tuple(shape);self.dtype=dtype;self.contiguous=contiguous
    def dim(self):return len(self.shape)
    def is_contiguous(self):return self.contiguous
    def float(self):return Tensor(self.shape)
    def chunk(self,n,dim=-1):
        assert n==2 and dim==-1 and self.shape[-1]%2==0;events.append('chunk');return (Tensor((self.shape[0],self.shape[1]//2)),)*2
    def __add__(self,other):events.append('materialized_add');return Tensor(self.shape)
    def __getitem__(self,key):return Tensor(tuple(min(s.stop or n,n) for s,n in zip(key,self.shape)),self.dtype)
def empty(shape,**kw):return Tensor(shape,kw['dtype'])
def jit(*args):
    events.append(('jit_args',args));return types.SimpleNamespace(fused_muse_spark_routing=lambda *a:events.append(('launch_shape',a[0].shape,a[1].shape)))
ns={'torch':types.SimpleNamespace(empty=empty,float32='float32',int32='int32'),'_jit_module':jit,'_NORMALIZATION_MODES':{'sigmoid':0,'softmax':1},'_LARGE_BATCH_TOKENS':2048}
exec(compile(ast.Module(body=ast.parse('from __future__ import annotations').body+nodes,type_ignores=[]),str(source),'exec'),ns)
checks=[]
for label,shape,contiguous,sum_halves in [('default',(512,256),True,False),('fused',(512,512),True,True),('fallback',(512,512),False,True),('large',(4096,512),True,True),('empty',(0,512),True,True)]:
    events.clear();out=ns['fused_muse_spark_routing_jit'](Tensor(shape,contiguous=contiguous),Tensor((256,)),8,1.,sum_halves=sum_halves)
    expected_flag=sum_halves and contiguous
    args=next(e[1] for e in events if isinstance(e,tuple) and e[0]=='jit_args')
    assert args[-1]==expected_flag
    assert ('materialized_add' in events)==(sum_halves and not contiguous)
    assert out[0].shape==(shape[0],8) and out[1].shape==(shape[0],8)
    checks.append({'label':label,'passed':True,'events':list(events)})
for shape,flag in [((512,256),True),((512,512),False)]:
    try:ns['fused_muse_spark_routing_jit'](Tensor(shape),Tensor((256,)),8,1.,sum_halves=flag)
    except AssertionError:checks.append({'label':'bad_columns','shape':shape,'sum_halves':flag,'passed':True})
    else:raise AssertionError('Malformed accumulator accepted')
report={'passed':True,'scope':'Exact wrapper AST with CPU spies; no arithmetic, compiler or GPU evidence','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checks':checks}
(P/'CPU_WRAPPER_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(checks)}))
