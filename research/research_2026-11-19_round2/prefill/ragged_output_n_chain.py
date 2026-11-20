"""Real-weight ragged-M, full-K/output-N screen with original complete KV tail.

This single GPU reconstructs all eight shards for correctness. Timed candidate
charges one local FC and full-output device staging; NCCL is not measured.
"""
import argparse,ctypes as C,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace as S
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_2026-11-19/prefill'))
sys.path.insert(0,str(ROOT/'research_2026-11-19/decode'))
from real_row_chain import checkpoint
import real_coupled as rc
from dflash_row_screen import compare,BASE
from output_n_package.runtime_projection import runtime_lib,native_workspace,VendorPlan

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--rows',type=int,nargs='+',default=[512,640,751,752,753,754,768,1023,1024,1535,3011,3012]);parser.add_argument('--rounds',type=int,default=4)
    args=parser.parse_args();torch.set_num_threads(1)
    assert hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest()==BASE
    (weight,kv,norm,knorm),receipt=checkpoint();print('real checkpoint verified',flush=True)
    native=C.CDLL(str(ROOT/'research_2026-11-18_lt_audit/rms_linux.so')).run_bf16_rms;ptr=C.c_void_p
    native.argtypes=[ptr,ptr,ptr,C.c_int,C.c_int,C.c_float,ptr];native.restype=C.c_int
    def normalize(x):
        y=torch.empty_like(x)
        assert native(x.data_ptr(),norm.data_ptr(),y.data_ptr(),x.shape[0],8192,1e-5,torch.cuda.current_stream().cuda_stream)==0
        return y
    path=ROOT/'sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py'
    spec=importlib.util.spec_from_file_location('ragged_n_kv',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    helper=object.__new__(module.FusedKVMaterializeHelper)
    fields=dict(device=torch.device('cuda'),n_layers=5,flat_kv_weight_t=kv,layer_out_dim=256,num_kv_heads=1,head_dim=128,rotary_dim=128,
        k_norm_weights=knorm,eps_values=torch.full((5,),1e-5,device='cuda'),_workspace_capacity=0,_workspace_dtype=None,_proj_workspace=None,
        _k_workspace=None,_v_workspace=None,_mm_out_supported=True,max_position_hint=131071,
        rotary_emb=S(cos_sin_cache=rc.configured_rope(torch.device('cuda'),500000)),_reserved_rope_cache_len=131072)
    for name,value in fields.items():setattr(helper,name,value)
    helper._ensure_workspace(max(args.rows),torch.bfloat16)
    lib=runtime_lib();backend,workspace=native_workspace();scratch=torch.empty(workspace,device='cuda',dtype=torch.uint8)
    generator=torch.Generator(device='cuda').manual_seed(args.seed)
    report=dict(complete=False,correct=True,seed=args.seed,parent_sha256=BASE,checkpoint=receipt,backend=backend,workspace_bytes=workspace,cases=[],
        scope='Actual FC and rank0 distinct-head KV/norm. Eight output-N shards checked; native full-M norm/five KV/RoPE/sparse writes. Timings include one local FC and explicit full output staging copy, exclude NCCL and peer compute. Not serving speed.',
        on_demand_scope='Qualification uses one existing live input and original full output, compares eight local shards sequentially; no second input allocation. Actual TP publish/gather tests remain required.')
    output=HERE/args.out
    def save():output.write_text(json.dumps(report,indent=2)+'\n')
    for m in args.rows:
        x=torch.randn((m,40960),device='cuda',dtype=torch.bfloat16,generator=generator)
        # One native warmup makes library/context initialization common to all arms.
        F.linear(x,weight);torch.cuda.synchronize()
        native_start=time.perf_counter_ns();reference=F.linear(x,weight);torch.cuda.synchronize();native_us=(time.perf_counter_ns()-native_start)/1000
        local=torch.empty((m,1024),device='cuda',dtype=torch.bfloat16)
        started=time.perf_counter_ns();handle=lib.projection_prepare(m,8192,40960,8,workspace)
        item=dict(rows=m,qualified=False,plans=[],native_fc_us=native_us,times_us={name:[] for name in ('baseline','candidate','control')},comparisons=[],
            ring_bytes_per_rank=7*m*1024*2,output_n_gather_layout_extra_copy_bytes=m*8192*2,
            conservative_transport_calibration_bytes=m*(6*8192+2*1024)+2*workspace+(16<<20))
        item['under128MiB_transport_calibration']=item['conservative_transport_calibration_bytes']<=128*2**20
        report['cases'].append(item);save()
        if not handle:item['reason']='vendor exposes no full-parent/local-N plan';save();del x,reference,local;continue
        plan=VendorPlan(lib,handle,8,8192,m,torch.device('cuda'));plan.k=40960
        for index in range(plan.choices()):
            info=plan.details(index);good=True
            for rank in range(8):
                status=plan.invoke(index,False,weight[rank*1024:(rank+1)*1024],x,local,scratch)
                if status or not torch.equal(local.view(torch.int16),reference[:,rank*1024:(rank+1)*1024].view(torch.int16)):
                    good=False;break
            item['plans'].append(dict(**info,exact=good))
            if good:plan.index=index;item['qualified']=True;item['chosen']=info;break
        torch.cuda.synchronize();item['first_use_extra_qualification_us']=(time.perf_counter_ns()-started)/1000
        if not item['qualified']:plan.close();save();del x,reference,local;continue
        assembled=torch.empty_like(reference);gathered=torch.empty_like(reference);capacity=m+256
        pools={arm:torch.full((2,5,capacity,1,128),-11.,device='cuda',dtype=torch.bfloat16) for arm in ('baseline','candidate','control')}
        locations=torch.randperm(capacity,device='cuda',generator=generator)[:m]
        cut=m//3;positions=torch.cat((torch.arange(cut,device='cuda')+65536,torch.arange(m-cut,device='cuda')+1003))
        def reconstruct():
            for rank in range(8):
                assert plan.invoke(plan.index,False,weight[rank*1024:(rank+1)*1024],x,local,scratch)==0
                assembled[:,rank*1024:(rank+1)*1024].copy_(local)
        def tail(fc,pool):
            normalized=normalize(fc)
            helper.materialize(normalized,positions,lambda layer,k,v:(pool[0,layer].index_copy_(0,locations,k),pool[1,layer].index_copy_(0,locations,v)))
        def run(arm):
            if arm=='candidate':
                assert plan.invoke(plan.index,False,weight[:1024],x,local,scratch)==0
                gathered.copy_(assembled);gathered[:,:1024].copy_(local);tail(gathered,pools[arm])
            else:tail(F.linear(x,weight),pools[arm])
        for iteration in range(-1,args.rounds):
            style='warmup' if iteration<0 else ('negated_perturbed','magnitude','integer','fresh_random')[iteration%4]
            if style=='negated_perturbed':x.neg_();x[:,::97].add_(.00390625)
            elif style=='magnitude':x[:,::5].mul_(8)
            elif style=='integer':x.random_(-2,3,generator=generator)
            elif style=='fresh_random':x.normal_(generator=generator)
            reconstruct();reference=F.linear(x,weight);check=compare(assembled,reference)
            if check['mismatches']:report['correct']=False;item['failure']=dict(style=style,fc=check);save();break
            order=['baseline','candidate','control'] if iteration%2==0 else ['control','candidate','baseline']
            for arm in order:
                torch.cuda.synchronize();start=time.perf_counter_ns();run(arm);torch.cuda.synchronize()
                if iteration>=0:item['times_us'][arm].append((time.perf_counter_ns()-start)/1000)
            checks={arm:compare(pools[arm].flatten(0,2),pools['baseline'].flatten(0,2)) for arm in ('candidate','control')}
            item['comparisons'].append(dict(iteration=iteration,style=style,fc=check,backing=checks))
            if any(v['mismatches'] for v in checks.values()):report['correct']=False;save();raise RuntimeError('KV backing mismatch')
        if report['correct']:
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run('candidate')
            x.neg_();reconstruct();graph.replay();run('baseline')
            item['changed_graph_backing']=compare(pools['candidate'].flatten(0,2),pools['baseline'].flatten(0,2));del graph
            report['correct'] &= item['changed_graph_backing']['mismatches']==0
            med={key:statistics.median(values) for key,values in item['times_us'].items()}
            saving=min(med['baseline'],med['control'])-med['candidate'];item['medians_us']=med
            item['first_use_extra_vs_ideal_saved_call']=item['first_use_extra_qualification_us']/saving if saving>0 else None
        print(m,'qualified',item['qualified'],'correct',report['correct'],'128MiB',item['under128MiB_transport_calibration'],flush=True);save()
        plan.close();del x,reference,local,assembled,gathered,pools
        if not report['correct']:break
    report['complete']=True;report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()

if __name__=='__main__':
    with torch.inference_mode():main()
