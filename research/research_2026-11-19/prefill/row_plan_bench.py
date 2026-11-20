"""Qualify full-parent vendor plans against full-K native row partitions."""
import argparse,ctypes as C,hashlib,json,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-18_lt_audit'))
from runtime_projection import native_workspace
from dflash_row_screen import compare,timing,BASE

def library():
    lib=C.CDLL(str(HERE/'row_plans.so'));p=C.c_void_p;i=C.c_int;u=C.c_uint64
    for name,args,result in [('make_row_plan',[i]*5+[u,i],p),('row_plan_count',[p],i),
        ('row_plan_attr',[p,i,i],i),('row_plan_call',[p,i,i,p,p,p,p,u,p],i),('row_plan_destroy',[p],None)]:
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=result
    return lib

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--rows',nargs='+',type=int,default=[8192,12288,15872,16384]);ap.add_argument('--stages',nargs='+',default=['kv','fc']);a=ap.parse_args()
    assert hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest()==BASE
    torch.set_num_threads(1);lib=library();workspace_info=native_workspace();assert workspace_info
    print('workspace',workspace_info,flush=True)
    backend,bytes_=workspace_info;scratch=torch.empty(bytes_,device='cuda',dtype=torch.uint8)
    gen=torch.Generator(device='cuda').manual_seed(a.seed)
    report=dict(seed=a.seed,baseline_sha256=BASE,native_backend=backend,workspace_bytes=bytes_,cases=[],complete=False,
                scope='Empirical full-parent cuBLASLt plan qualification; fixed fullK/fullN localM; exact native BF16 outputs; synthetic operands; no transport or score projection.')
    out=HERE/a.out
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    for stage in a.stages:
        n,k,tr=(1280,8192,0) if stage=='kv' else (8192,40960,1)
        w=torch.randn((n,k) if tr else (k,n),device='cuda',dtype=torch.bfloat16,generator=gen)/k**.5
        for m in a.rows:
            assert m%8==0
            x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16,generator=gen)
            native=lambda z:F.linear(z,w) if tr else torch.mm(z,w)
            ref=native(x);local=torch.empty((m//8,n),device='cuda',dtype=torch.bfloat16);full=torch.empty_like(ref)
            plan=lib.make_row_plan(m,m//8,n,k,tr,bytes_,16)
            item=dict(stage=stage,rows=m,available=lib.row_plan_count(plan) if plan else 0,plans=[],chosen=None)
            report['cases'].append(item);save()
            if not plan:continue
            def call(index,fullshape,z,y):
                status=lib.row_plan_call(plan,index,int(fullshape),w.data_ptr(),z.data_ptr(),y.data_ptr(),scratch.data_ptr(),bytes_,torch.cuda.current_stream().cuda_stream)
                assert status==0, status
                return y
            for index in range(lib.row_plan_count(plan)):
                info=dict(index=index,attrs=[lib.row_plan_attr(plan,index,f) for f in range(9)])
                item['plans'].append(info)
                try:
                    call(index,True,x,full);info['full']=compare(full,ref)
                    if info['full']['mismatches']:continue
                    matches=[]
                    for start in range(0,m,m//8):
                        call(index,False,x[start:start+m//8],local)
                        matches.append(compare(local,ref[start:start+m//8])['mismatches'])
                    info['local_mismatches']=matches
                    if any(matches):continue
                    item['chosen']=index;break
                except AssertionError as error:info['error']=str(error)
            if item['chosen'] is not None:
                index=item['chosen'];checks=[]
                for style in ('negated','magnitude','integer'):
                    if style=='negated':x.neg_()
                    elif style=='magnitude':x[:,::5]*=8
                    else:x.random_(-2,3,generator=gen)
                    ref=native(x)
                    bad=[]
                    for start in range(0,m,m//8):
                        call(index,False,x[start:start+m//8],local)
                        bad.append(compare(local,ref[start:start+m//8])['mismatches'])
                    checks.append(dict(style=style,rank_mismatches=bad))
                item['changed_input_checks']=checks
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):call(index,False,x[:m//8],local)
                x.neg_();graph.replay();ref=native(x)
                item['graph_changed_input']=compare(local,ref[:m//8]);del graph
                item['qualified']=not any(any(c['rank_mismatches']) for c in checks) and not item['graph_changed_input']['mismatches']
            print(stage,m,'plans',item['available'],'chosen',item['chosen'],'qualified',item.get('qualified'),flush=True)
            save();lib.row_plan_destroy(plan);del x,ref,local,full
        del w
    report['complete']=True;report['peak_mib']=torch.cuda.max_memory_allocated()/2**20;save()

if __name__=='__main__':
    with torch.inference_mode():main()
