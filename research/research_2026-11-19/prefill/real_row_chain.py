"""Real checkpoint row-FC projection plus unchanged full-M normalization and KV.

Candidate timing is a local-rank compute/copy envelope, NOT eight-GPU elapsed
time. All other rank FC rows are freshly reconstructed outside timing, then a
full-size device copy explicitly charges gather staging. Network is excluded.
"""
import argparse,ctypes as C,hashlib,importlib.util,inspect,json,sys,time
from pathlib import Path
from types import SimpleNamespace as S
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-18_lt_audit'))
import real_coupled as rc
from runtime_projection import native_workspace
from row_plan_bench import library
from dflash_row_screen import compare,BASE

def checkpoint():
    source=inspect.getsource(rc.checkpoint).replace('"public_fc_receipt.json"','"recovered_public_fc_receipt.json"').replace('"public_aux_receipt.json"','"recovered_public_aux_receipt.json"')
    scope=dict(rc.__dict__,SOURCE=HERE);exec(compile(source,'recovered_checkpoint','exec'),scope)
    return scope['checkpoint'](torch.device('cuda'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True)
    p.add_argument('--rows',nargs='+',type=int,default=[512,768,3072,8192,16384]);p.add_argument('--rounds',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(1)
    assert hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest()==BASE
    (weight,kv,norm,knorm),receipt=checkpoint();print('checkpoint verified',receipt['revision'],flush=True)
    gen=torch.Generator(device='cuda').manual_seed(args.seed)
    native=C.CDLL(str(ROOT/'research_2026-11-18_lt_audit/rms_linux.so')).run_bf16_rms;ptr=C.c_void_p
    native.argtypes=[ptr,ptr,ptr,C.c_int,C.c_int,C.c_float,ptr];native.restype=C.c_int
    def normalize(x):
        y=torch.empty_like(x)
        assert native(x.data_ptr(),norm.data_ptr(),y.data_ptr(),x.shape[0],8192,1e-5,torch.cuda.current_stream().cuda_stream)==0
        return y
    path=ROOT/'sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py'
    spec=importlib.util.spec_from_file_location('real_row_fused',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    h=object.__new__(module.FusedKVMaterializeHelper)
    fields=dict(device=torch.device('cuda'),n_layers=5,flat_kv_weight_t=kv,layer_out_dim=256,num_kv_heads=1,head_dim=128,rotary_dim=128,
                k_norm_weights=knorm,eps_values=torch.full((5,),1e-5,device='cuda'),_workspace_capacity=0,_workspace_dtype=None,
                _proj_workspace=None,_k_workspace=None,_v_workspace=None,_mm_out_supported=True,max_position_hint=131071,
                rotary_emb=S(cos_sin_cache=rc.configured_rope(torch.device('cuda'),500000)),_reserved_rope_cache_len=131072)
    for k,v in fields.items():setattr(h,k,v)
    h._ensure_workspace(max(args.rows),torch.bfloat16)
    lib=library();backend,ws=native_workspace();scratch=torch.empty(ws,device='cuda',dtype=torch.uint8)
    report=dict(seed=args.seed,baseline_sha256=BASE,complete=False,correct=True,checkpoint=receipt,cases=[],
                scope='Real public FC and rank0 own KVhead; full native RMS and five KV/RoPE/scatter stages. All8 FCrow partitions reconstructed. Timed candidate includes one rank FC plus full output copy and full native tail; excludes TP8 collective. No serving claim.',
                head_ownership='Original rank0 head is retained; no packedKV gather, no other-rank head substitution.',
                transport_model='TP8 ring rawFC all-gather sends7/8*M*8192*2 bytes/rank.')
    out=HERE/args.out
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    for m in args.rows:
        assert m%8==0;local_m=m//8
        x=torch.randn((m,40960),device='cuda',dtype=torch.bfloat16,generator=gen)
        reference=F.linear(x,weight);assembled=torch.empty_like(reference);gathered=torch.empty_like(reference);local=torch.empty((local_m,8192),device='cuda',dtype=torch.bfloat16)
        plan=lib.make_row_plan(m,local_m,8192,40960,1,ws,16);assert plan
        item=dict(rows=m,plans=[],chosen=None,comparisons=[],times_us={a:[] for a in ('baseline','candidate','control')},
                  ring_bytes_per_rank=7*local_m*8192*2,explicit_gather_copy_bytes=m*8192*2)
        report['cases'].append(item);save()
        def call(index,full,z,y):
            assert lib.row_plan_call(plan,index,int(full),weight.data_ptr(),z.data_ptr(),y.data_ptr(),scratch.data_ptr(),ws,torch.cuda.current_stream().cuda_stream)==0
            return y
        for index in range(lib.row_plan_count(plan)):
            call(index,True,x,gathered);info=dict(index=index,attrs=[lib.row_plan_attr(plan,index,f) for f in range(9)],full=compare(gathered,reference));item['plans'].append(info)
            if info['full']['mismatches']:continue
            for start in range(0,m,local_m):call(index,False,x[start:start+local_m],assembled[start:start+local_m])
            info['all_rows']=compare(assembled,reference)
            if not info['all_rows']['mismatches']:item['chosen']=index;break
        if item['chosen'] is None:
            item['fallback']='No exact native-equivalent plan found';save();lib.row_plan_destroy(plan);continue
        chosen=item['chosen'];capacity=m+256
        pools={arm:torch.full((2,5,capacity,1,128),-11.,device='cuda',dtype=torch.bfloat16) for arm in ('baseline','candidate','control')}
        locations=torch.randperm(capacity,device='cuda',generator=gen)[:m]
        split=m//3;positions=torch.cat((torch.arange(split,device='cuda')+65536,torch.arange(m-split,device='cuda')+1003))
        def tail(fc,pool):
            normalized=normalize(fc)
            h.materialize(normalized,positions,lambda layer,k,v:(pool[0,layer].index_copy_(0,locations,k),pool[1,layer].index_copy_(0,locations,v)))
        def run(arm):
            if arm=='candidate':
                call(chosen,False,x[:local_m],local)
                gathered.copy_(assembled);gathered[:local_m].copy_(local)
                tail(gathered,pools[arm])
            else:tail(F.linear(x,weight),pools[arm])
        for iteration in range(-1,args.rounds):
            style=('warmup' if iteration<0 else ('negated_perturbed','magnitude','integer','fresh_random')[iteration%4])
            if style=='negated_perturbed':x.neg_();x[:,::97].add_(.00390625)
            elif style=='magnitude':x[:,::5].mul_(8)
            elif style=='integer':x.random_(-2,3,generator=gen)
            elif style=='fresh_random':x.normal_(generator=gen)
            for start in range(0,m,local_m):call(chosen,False,x[start:start+local_m],assembled[start:start+local_m])
            reference=F.linear(x,weight);fc_check=compare(assembled,reference)
            if fc_check['mismatches']:
                report['correct']=False;item['comparisons'].append(dict(iteration=iteration,fc=fc_check));save();raise RuntimeError('FC mismatch')
            order=['baseline','candidate','control']
            if iteration>=0 and iteration%2:order.reverse()
            for arm in order:
                torch.cuda.synchronize();start=time.perf_counter_ns();run(arm);torch.cuda.synchronize();us=(time.perf_counter_ns()-start)/1000
                if iteration>=0:item['times_us'][arm].append(us)
            comparisons={arm:compare(pools[arm].flatten(0,2),pools['baseline'].flatten(0,2)) for arm in ('candidate','control')}
            item['comparisons'].append(dict(iteration=iteration,input_style=style,fc=fc_check,backing_pools=comparisons))
            if any(v['mismatches'] for v in comparisons.values()):report['correct']=False;save();raise RuntimeError('KV backing mismatch')
            save()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):run('candidate')
        x.neg_()
        for start in range(0,m,local_m):call(chosen,False,x[start:start+local_m],assembled[start:start+local_m])
        graph.replay();run('baseline')
        item['changed_graph_backing']=compare(pools['candidate'].flatten(0,2),pools['baseline'].flatten(0,2))
        report['correct'] &= item['changed_graph_backing']['mismatches']==0
        print(m,'real chain exact',report['correct'],flush=True);save();del graph
        lib.row_plan_destroy(plan);del x,reference,assembled,gathered,local,pools
    report['complete']=True;report['peak_mib']=torch.cuda.max_memory_allocated()/2**20;save()

if __name__=='__main__':
    with torch.inference_mode():main()
