"""Actual exported adapters; single-GPU virtual peers, explicitly not NCCL."""
import argparse, copy, fcntl, hashlib, importlib, importlib.util, json, pathlib, resource, statistics, sys, time, types
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=pathlib.Path('/mnt/d/OpenKernelChallenge/serving');O=R/'research_2026-11-19_final'
SOURCE=O/'assembled/python/sglang'
for name,path in [('sglang',SOURCE),('sglang.srt',SOURCE/'srt'),('sglang.srt.speculative',SOURCE/'srt/speculative')]:
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
import torch
import torch.distributed as dist
torch.set_num_threads(1)
row=importlib.import_module('sglang.srt.speculative.dflash_projection_row')
nmod=importlib.import_module('sglang.srt.speculative.dflash_projection_n.adapter')
build=importlib.import_module('sglang.srt.speculative.dflash_projection_n.native_build')
sys.path.insert(0,str(R/'research_2026-11-19/prefill'))
import real_row_chain

def main(seed):
    def gather_objects(out,value,group=None):out[:]=[copy.deepcopy(value) for _ in out]
    dist.all_gather_object=gather_objects;dist.all_reduce=lambda value,**kwargs:None
    (weight,kv,norm,knorm),receipt=real_row_chain.checkpoint()
    linear=torch.nn.Linear(weight.shape[1],weight.shape[0],bias=False,device='meta',dtype=weight.dtype)
    linear.weight=torch.nn.Parameter(weight,requires_grad=False)
    build.build_native();row._library()
    gen=torch.Generator(device='cuda').manual_seed(seed)
    class Group:
        world_size=8;rank_in_group=0;cpu_group=None
        peers=None
        def all_gather(self,local,dim):return torch.cat([local,*self.peers[1:]],dim=dim)
    report={'scope':'Actual exported CUDA adapters on one SM86 GPU; CPU votes mocked, eight peer results recomputed serially outside timing. No NCCL, full server or H200 emulation.',
            'seed':seed,'weights':receipt,'cases':[],'complete':False,'correct':True}
    out=O/f'adapter_probe_{seed}.json'
    def save():out.write_text(json.dumps(report,indent=2))
    for m in [8,512,32,768,128,3072,256]:
        group=Group();isrow=m>=512
        router=row.RowProjectionRouter(linear,group,empirical=True) if isrow else nmod.OutputNRouter(linear,group,mode='qualified')
        assert router.prepare_registration([(m,40960)])[0]
        x=torch.randn((m,40960),device='cuda',dtype=torch.bfloat16,generator=gen)
        y=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=gen);y[:,::5]*=8
        qualification=router.register([x,y]);del y
        enabled=qualification.get('accepted',qualification.get('globally_enabled',False))
        case={'rows':m,'kind':'row' if isrow else 'output_n','enabled':enabled,'qualification':qualification,'mismatches':[],
              'times_us':{name:[] for name in ('baseline','disabled','row_only','decode_only','joint','control')}}
        report['cases'].append(case);save()
        if not enabled:
            case['fallback_exact']=torch.equal(router(x),linear(x));save();continue
        def refresh():
            if isrow:
                plan=router.plans[(tuple(x.shape),row._stream(x))]
                group.peers=[plan.local(x,weight,r) for r in range(8)]
            else:
                entry=router.entries[tuple(x.shape)];group.peers=[]
                for rank in range(8):
                    group.rank_in_group=rank;group.peers.append(router._local(entry,x))
                group.rank_in_group=0
            assert all(p is not None for p in group.peers)
            assembled=torch.cat(group.peers,dim=0 if isrow else -1)
            count=int(torch.count_nonzero(assembled.view(torch.int16)!=linear(x).view(torch.int16)))
            case['mismatches'].append(count);assert count==0
        def invoke(name):
            active=name=='joint' or (isrow and name=='row_only') or (not isrow and name=='decode_only')
            return router(x) if active else linear(x)
        for it in range(-2,12):
            if it%3==0:x.normal_(generator=gen)
            elif it%3==1:x.neg_();x[:,::97].add_(.00390625)
            else:x.random_(-2,3,generator=gen)
            refresh()
            order=list(case['times_us']);order=order[it%6:]+order[:it%6]
            for name in order:
                torch.cuda.synchronize();t=time.perf_counter_ns();actual=invoke(name);torch.cuda.synchronize();dt=(time.perf_counter_ns()-t)/1000
                assert torch.equal(actual.view(torch.int16),linear(x).view(torch.int16))
                if it>=0:case['times_us'][name].append(dt)
                del actual
        # Capture and mutate both the local input and peer buffers in place.
        refresh();peers=group.peers
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):router(x)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):captured=router(x)
        x.neg_();refresh();fresh=group.peers
        for old,new in zip(peers,fresh):old.copy_(new)
        group.peers=peers;graph.replay();torch.cuda.synchronize()
        case['changed_graph_exact']=torch.equal(captured.view(torch.int16),linear(x).view(torch.int16));assert case['changed_graph_exact']
        case['medians_us']={k:statistics.median(v) for k,v in case['times_us'].items()}
        save();print(m,case['medians_us'],'graph exact',case['changed_graph_exact'],flush=True)
        del graph,captured
        router.close(graphs_destroyed=True) if isrow else router.retire(graphs_destroyed=True)
        del router,x,group,peers,fresh;torch.cuda.empty_cache()
    report['complete']=True;report['peak_allocated']=torch.cuda.max_memory_allocated();save()

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seed',type=int,default=19011931);args=a.parse_args()
    with (R/'research_2026-11-18_rl/.gpu.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        with torch.inference_mode():main(args.seed)
