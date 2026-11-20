"""Isolated exact-shape DFlash context append CUDA-graph experiment."""
import argparse,ast,contextlib,hashlib,importlib.util,json,random,statistics,sys,time,types
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;ROOT=P.parents[3];S=ROOT/'sglang/python/sglang'

def fingerprint(t):return (t.data_ptr(),tuple(t.shape),t.stride(),t.dtype,t.device)
class Bindings:
    def __init__(self,**tensors):
        self.tensors=tensors;self.saved={k:fingerprint(v) for k,v in tensors.items()}
    def validate(self,**current):
        if current.keys()!=self.tensors.keys():raise RuntimeError('binding names changed')
        for k,v in current.items():
            if fingerprint(v)!=self.saved[k]:raise RuntimeError('binding changed: '+k)

def extract(path,name,namespace,cls=None):
    tree=ast.parse(path.read_text());body=tree.body
    if cls:body=next(n.body for n in body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name);fn.decorator_list=[]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--out',type=Path,required=True);pa.add_argument('--cpu',action='store_true');pa.add_argument('--rows',type=int,nargs='+',default=[8,32,128,256]);pa.add_argument('--samples',type=int,default=10);a=pa.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True);checks=[]
    report={'status':'RUNNING','checks':checks,'rows':[],'scope':'Actual project_target_hidden method, nn.Linear BF16 full40960x8192, native sgl_kernel RMSNorm, stock five-layer fused KV helper and exact prefix writer; synthetic weights/cache/positions. No serving runtime, TP collective, scheduler or radix tree.'}
    def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
    def check(name,ok):checks.append({'name':name,'passed':bool(ok)});save();assert ok,name
    x=torch.ones(4,8);b=Bindings(x=x);b.validate(x=x);check('same_binding',True)
    original=x.view_as(x)
    for mutation in ('storage','shape','stride','dtype'):
        x.data=original
        if mutation=='storage':x.data=x.clone()
        elif mutation=='shape':x.resize_(2,16)
        elif mutation=='stride':x.as_strided_((4,8),(1,4))
        else:x.data=x.to(torch.float64)
        rejected=False
        try:b.validate(x=x)
        except RuntimeError:rejected=True
        check('reject_'+mutation,rejected)
        x.data=original;b.validate(x=x)
    check('restore_recovers',True)
    if a.cpu:report.update(status='PASS',check_count=len(checks));save();return
    import triton
    from sgl_kernel import rmsnorm
    torch.set_num_threads(1);torch.manual_seed(917)
    report['device']=torch.cuda.get_device_name();report['torch']=torch.__version__
    report['allow_bf16_reduced_precision_reduction']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    for name in ('sglang','sglang.srt','sglang.srt.utils'):
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    sys.modules['sglang.srt.utils'].is_cpu=lambda:False
    move=load('append_cache_move',S/'kernels/ops/kvcache/cache_move.py')
    fused=load('append_fused_kv',S/'kernels/ops/speculative/fused_kv_materialize.py')
    writer=extract(S/'srt/mem_cache/memory_pool.py','_set_kv_buffer_prefix_valid_impl',{'torch':torch,'triton':triton,'set_kv_buffer_prefix_valid_tiled':move.set_kv_buffer_prefix_valid_tiled})
    project=extract(S/'srt/models/dflash.py','project_target_hidden',{'torch':torch},'DFlashDraftModel')
    H=8192;D=128;L=5;N=max(a.rows);capacity=max(1024,N*4)
    weight=torch.randn(H,40960,dtype=torch.bfloat16,device='cuda')*.003
    fc=torch.nn.Linear(40960,H,bias=False,device='meta',dtype=torch.bfloat16);fc.weight=torch.nn.Parameter(weight,requires_grad=False)
    normweight=torch.ones(H,dtype=torch.bfloat16,device='cuda')
    model=types.SimpleNamespace(fc=fc,quantizable=False,num_context_features=5,config=types.SimpleNamespace(hidden_size=H),hidden_norm=lambda x:rmsnorm(x,normweight,1e-6))
    layers=[]
    for i in range(L):
        attn=types.SimpleNamespace(num_kv_heads=1,head_dim=D,q_size=0,kv_size=D,qkv_proj=types.SimpleNamespace(weight=torch.randn(2*D,H,device='cuda',dtype=torch.bfloat16)*.01),k_norm=types.SimpleNamespace(weight=torch.rand(D,device='cuda',dtype=torch.bfloat16)+.5,variance_epsilon=1e-6),rotary_emb=types.SimpleNamespace(rotary_dim=D,is_neox_style=True))
        layers.append(types.SimpleNamespace(self_attn=attn))
    angles=torch.randn(4096,D//2,device='cuda');rope=types.SimpleNamespace(rotary_dim=D,is_neox_style=True,cos_sin_cache=torch.cat((angles.cos(),angles.sin()),1))
    helper=fused.FusedKVMaterializeHelper(layers,rope,1,D,torch.device('cuda'),max_position_hint=4096);helper._ensure_workspace(N,torch.bfloat16)
    hidden=torch.randn(N,40960,device='cuda',dtype=torch.bfloat16)
    pos=torch.arange(N,device='cuda',dtype=torch.int64);loc=torch.arange(N,device='cuda',dtype=torch.int64).view(-1,8);comm=torch.full((N//8,),8,device='cuda',dtype=torch.int32)
    banks=[(torch.full((L,capacity,1,D),-91,dtype=torch.bfloat16,device='cuda'),torch.full((L,capacity,1,D),-73,dtype=torch.bfloat16,device='cuda')) for _ in range(2)]
    immutable=Bindings(hidden=hidden,weight=weight,norm=normweight,kv=helper.flat_kv_weight_t,k0=banks[0][0],k1=banks[1][0])
    def body(rows,bank):
        ctx=project(model,hidden[:rows])
        def write(i,k,v):writer(k,v,banks[bank][0][i],banks[bank][1][i],loc[:rows//8],comm[:rows//8],D,torch.bfloat16)
        helper.materialize(ctx,pos[:rows],write)
        return ctx
    def bitwise(x,y):return torch.equal(x.view(torch.uint8),y.view(torch.uint8))
    def timed(call):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();t=time.perf_counter();start.record();call();end.record();end.synchronize()
        return {'wall_us':(time.perf_counter()-t)*1e6,'gpu_us':start.elapsed_time(end)*1000}
    graph_refs=[]
    for rows in a.rows:
        before=torch.cuda.memory_allocated();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):body(rows,1)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream),torch.inference_mode():captured=body(rows,1)
        graph_refs.append((graph,captured));extra=torch.cuda.memory_allocated()-before
        for generation in range(5):
            hidden.mul_(-.5).add_(.03125*(generation+1))
            pos.copy_((torch.arange(N,device='cuda')*7+generation*31)%4096)
            # Recycled physical slots, reordered requests, new Tensor sources every generation.
            freshloc=((torch.arange(N,device='cuda')*3+generation*29)%capacity).reshape(-1,8)
            freshcomm=((torch.arange(N//8,device='cuda')+generation*3)%9).to(torch.int32)
            loc.copy_(freshloc);comm.copy_(freshcomm)
            for kb,vb in banks:kb.fill_(-91);vb.fill_(-73)
            with torch.inference_mode():expected=body(rows,0)
            graph.replay();torch.cuda.synchronize()
            check('rows%d_generation%d_ctx'%(rows,generation),bitwise(expected,captured))
            check('rows%d_generation%d_kv_untouched_slots'%(rows,generation),all(bitwise(x,y) for x,y in zip(banks[0],banks[1])))
        immutable.validate(hidden=hidden,weight=weight,norm=normweight,kv=helper.flat_kv_weight_t,k0=banks[0][0],k1=banks[1][0])
        # Decisive stale-location negative control: different reference location must disagree.
        banks[0][0].fill_(-91);banks[1][0].fill_(-91);comm.fill_(8)
        loc.copy_(torch.arange(N,device='cuda').reshape(-1,8));graph.replay()
        loc.add_(capacity//2)
        with torch.inference_mode():body(rows,0)
        check('rows%d_stale_location_negative'%rows,not bitwise(banks[0][0],banks[1][0]))
        source_hidden=hidden[:rows].clone();source_pos=pos[:rows].clone();source_loc=loc[:rows//8].clone();source_comm=comm[:rows//8].clone()
        def staged(copy_hidden):
            if copy_hidden:hidden[:rows].copy_(source_hidden)
            pos[:rows].copy_(source_pos);loc[:rows//8].copy_(source_loc);comm[:rows//8].copy_(source_comm);graph.replay()
        calls={'eagerA':lambda:body(rows,1),'graph':graph.replay,'graph_metadata':lambda:staged(False),'graph_fullstaging':lambda:staged(True),'eagerB':lambda:body(rows,1)}
        timings={k:[] for k in calls};rng=random.Random(rows)
        with torch.inference_mode():
            for sample in range(a.samples):
                order=list(timings);rng.shuffle(order)
                for key in order:timings[key].append(timed(calls[key]))
        med={k:{metric:statistics.median(v[metric] for v in values) for metric in ('wall_us','gpu_us')} for k,values in timings.items()}
        report['rows'].append({'rows':rows,'timings':timings,'medians':med,'graph_extra_allocated_bytes':extra,'graph_vs_eager_wall_ratio':med['eagerA']['wall_us']/med['graph']['wall_us'],'AA_wall_ratio':med['eagerA']['wall_us']/med['eagerB']['wall_us']});save()
    report.update(status='PASS',check_count=len(checks),max_allocated_bytes=torch.cuda.max_memory_allocated());save()
if __name__=='__main__':main()
