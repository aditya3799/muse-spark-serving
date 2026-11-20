"""CPU wrapper metadata and fallback contracts; no arithmetic or GPU claims."""
import importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
from runtime_imports import prepare
prepare()
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fusion_candidate',P/'norm_residual_aux.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[];calls=[]
def check(name,ok):
    assert ok,name
    checks.append({'name':name,'passed':bool(ok)})
class Module:
    def __getattr__(self,name):
        def fn(branch,stream,output,new_stream,router,post,pre,alpha,beta,ep1,ep2,pbf16,aux,slot):
            output.copy_(branch);new_stream.copy_(stream)
            if aux is not None:aux[:,slot*branch.shape[1]:(slot+1)*branch.shape[1]].copy_(stream)
            calls.append({'entry':name,'slot':slot,'fused':aux is not None})
        return fn
m._jit_module=lambda *args:Module()
b=torch.zeros((3,8192),dtype=torch.bfloat16);s=torch.arange(3*8192,dtype=torch.float32).view_as(b)
for cluster in [1,2,4,8,16]:
    for slot in [0,2,4]:
        out=torch.full((3,8192*5),-1,dtype=torch.bfloat16)
        m.fused_norm_residual(b,s,1e-6,1e-6,cluster_size=cluster,aux_out=out,aux_slot=slot)
        check('cluster%d_slot%d_value'%(cluster,slot),torch.equal(out[:,slot*8192:(slot+1)*8192],s.to(torch.bfloat16)))
        check('cluster%d_slot%d_dispatch'%(cluster,slot),calls[-1]['fused']==(cluster==1))
valid=torch.empty((3,8192*5),dtype=torch.bfloat16)
for reason,out,slot in [('dtype',valid.float(),0),('rows',valid[:2],0),('width',valid[:,:-1],0),('stride',valid.t(),0),('negative',valid,-1),('end',valid,5),('bool',valid,True),('float',valid,1.0)]:
    try:m.fused_norm_residual(b,s,1e-6,1e-6,cluster_size=1,aux_out=out,aux_slot=slot)
    except ValueError:check('reject_'+reason,True)
    else:check('reject_'+reason,False)
# Exact launch-policy preservation at the single-CTA fusion boundary.
for rows in [1,17,191,192,256,1024,16384]:
    with patch.object(torch.cuda,'get_device_capability',lambda device:(9,0)):
        selected=[]
        m._jit_module=lambda *args:(selected.append(args) or Module())
        bx=b[:1].expand(rows,-1).contiguous();sx=s[:1].expand(rows,-1).contiguous()
        m.fused_norm_residual(bx,sx,1e-6,1e-6,aux_out=torch.empty(rows,8192,dtype=torch.bfloat16))
        check('h200_cluster1_m%d'%rows,selected[0][3]==1 and selected[0][1:3]==m._launch_config(8192,rows))
(P/'cpu_contract_results.json').write_text(json.dumps({'status':'PASS','check_count':len(checks),'checks':checks,'scope':'CPU mock launch contracts; real wrapper, no CUDA compilation.'},indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)}))
