"""Actual exported projection + source-extracted worker/model hooks + real KV helper.
Single GPU with live virtual peers. No serving throughput measurement.
"""
import argparse,ast,contextlib,ctypes as C,hashlib,importlib.util,json,logging,types
from typing import Optional
import adapter_probe as ap
import torch
R=ap.R;O=R/'research_2026-11-20/local_validation'
def method(path,name):
    node=next(n for cls in ast.parse(path.read_text()).body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    ns=dict(torch=torch,Optional=Optional,draft_tp_context=lambda _:contextlib.nullcontext(),logger=logging.getLogger('gpu-hooks'))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]
def main(m):
    ap.dist.all_gather_object=lambda out,value,**kwargs:out.__setitem__(slice(None),[value]*len(out))
    ap.dist.all_reduce=lambda *a,**k:None
    (weight,kv,norm,knorm),receipt=ap.real_row_chain.checkpoint()
    linear=torch.nn.Linear(40960,8192,bias=False,device='meta',dtype=torch.bfloat16)
    linear.weight=torch.nn.Parameter(weight,requires_grad=False)
    gen=torch.Generator(device='cuda').manual_seed(20112071+m)
    class Group:
        world_size=8;rank_in_group=0;cpu_group=None;peers=None
        def all_gather(self,local,dim):return torch.cat([local,*self.peers[1:]],dim=dim)
    group=Group();isrow=m>=512
    router=ap.row.RowProjectionRouter(linear,group,empirical=True) if isrow else ap.nmod.OutputNRouter(linear,group,mode='qualified')
    assert router.prepare_registration([(m,40960)])[0]
    x=torch.randn((m+8,40960),device='cuda',dtype=torch.bfloat16,generator=gen)
    q=router.register([x[:m],torch.randn_like(x[:m])]);assert q.get('accepted',q.get('globally_enabled',False)),q
    target=router.plans[((m,40960),ap.row._stream(x))] if isrow else router
    method_name='local' if isrow else '_local'
    saved_local=getattr(target,method_name)
    fast_calls=[]
    def require_fast(*args,**kwargs):
        result=saved_local(*args,**kwargs)
        assert result is not None,'unexpected fallback in composed fast-path test'
        fast_calls.append(True)
        return result
    setattr(target,method_name,require_fast)
    def refresh():
        if isrow:
            plan=router.plans[((m,40960),ap.row._stream(x))]
            fresh=[plan.local(x[:m],weight,r) for r in range(8)]
        else:
            fresh=[]
            for r in range(8):
                group.rank_in_group=r;fresh.append(router._local(router.entries[(m,40960)],x[:m]))
            group.rank_in_group=0
        assert all(p is not None for p in fresh)
        assert torch.equal(torch.cat(fresh,dim=0 if isrow else 1).view(torch.int16),linear(x[:m]).view(torch.int16))
        if group.peers is None:group.peers=fresh
        else:
            for old,new in zip(group.peers,fresh):old.copy_(new)
    native=C.CDLL(str(R/'research_2026-11-18_lt_audit/rms_linux.so')).run_bf16_rms
    native.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_float,C.c_void_p];native.restype=C.c_int
    def normalize(z):
        out=torch.empty_like(z)
        assert native(z.data_ptr(),norm.data_ptr(),out.data_ptr(),m,8192,1e-5,torch.cuda.current_stream().cuda_stream)==0
        return out
    mp=ap.SOURCE/'srt/models/dflash.py';wp=ap.SOURCE/'srt/speculative/dflash_worker_v2.py'
    model=types.SimpleNamespace(fc=linear,quantizable=False,hidden_norm=normalize,num_context_features=5,config=types.SimpleNamespace(hidden_size=8192))
    model.project_target_hidden=types.MethodType(method(mp,'project_target_hidden'),model)
    append=method(wp,'_append_target_hidden_to_draft_kv_by_loc')
    hp=R/'sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py'
    spec=importlib.util.spec_from_file_location('exported_chain_kv',hp);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    h=object.__new__(mod.FusedKVMaterializeHelper)
    fields=dict(device=torch.device('cuda'),n_layers=5,flat_kv_weight_t=kv,layer_out_dim=256,num_kv_heads=1,head_dim=128,rotary_dim=128,k_norm_weights=knorm,eps_values=torch.full((5,),1e-5,device='cuda'),_workspace_capacity=0,_workspace_dtype=None,_proj_workspace=None,_k_workspace=None,_v_workspace=None,_mm_out_supported=True,max_position_hint=131071,rotary_emb=types.SimpleNamespace(cos_sin_cache=ap.real_row_chain.rc.configured_rope(torch.device('cuda'),500000)),_reserved_rope_cache_len=131072)
    for key,val in fields.items():setattr(h,key,val)
    h._ensure_workspace(m,torch.bfloat16)
    pools=[torch.full((2,5,m+256,1,128),-11.,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    loc=torch.randperm(m+256,device='cuda',generator=gen)[:m];pos=torch.arange(m,device='cuda')+65536
    workers=[]
    for index in range(2):
        w=types.SimpleNamespace(draft_model=model,model_runner=types.SimpleNamespace(device=x.device),draft_parallel_layout=None,lilicorr=None,_use_fused_kv_materialize=False,_fused_kv_helper=None,_prefill_row_projection=router if index and isrow else None,_decode_n_projection=router if index and not isrow else None)
        def write(*,ctx_hidden,ctx_cache_loc,ctx_positions,pool=pools[index],**kwargs):
            h.materialize(ctx_hidden,ctx_positions,lambda layer,k,v:(pool[0,layer].index_copy_(0,ctx_cache_loc,k),pool[1,layer].index_copy_(0,ctx_cache_loc,v)))
        w._append_target_hidden_sequential=write;workers.append(w)
    def run(i):append(workers[i],target_hidden=x,cache_loc=loc,positions=pos,prefill_projection=isrow)
    checks=[]
    for it in range(4):
        if it==1:x.random_(-2,3,generator=gen)
        else:x.normal_(generator=gen);x[:,::7].mul_(8 if it==2 else 1)
        refresh();run(0);run(1)
        assert torch.equal(pools[0].view(torch.int16),pools[1].view(torch.int16))
        checks.append('changed_input_'+str(it))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):run(1)
    x.neg_();refresh();graph.replay();run(0);torch.cuda.synchronize()
    assert torch.equal(pools[0].view(torch.int16),pools[1].view(torch.int16))
    checks.append('changed_graph_full_backing_equal')
    del graph
    router.close(graphs_destroyed=True) if isrow else router.retire(graphs_destroyed=True)
    result=dict(complete=True,correct=True,rows=m,checks=checks,scope=__doc__,qualification=q,peak_allocated=torch.cuda.max_memory_allocated(),sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (mp,wp,hp)})
    (O/f'exported_chain_{m}.json').write_text(json.dumps(result,indent=2));print('PASS exported chain',m,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,required=True);args=p.parse_args()
    with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):main(args.rows)
