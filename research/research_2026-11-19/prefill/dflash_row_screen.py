"""Exact stage screen for TP-row sharding of replicated draft KV materialization.

Local rank time excludes all-gather. All rank outputs are independently computed
and reassembled before timing; the native full-width projection is the baseline.
"""
import argparse, ctypes, hashlib, importlib.util, json, statistics, sys
from pathlib import Path
from types import SimpleNamespace as S
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE='8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b'

def compare(a,b):
    wrong=0;maximum=0.
    flat_a,flat_b=a.reshape(-1),b.reshape(-1)
    for start in range(0,flat_a.numel(),1<<20):
        x,y=flat_a[start:start+(1<<20)],flat_b[start:start+(1<<20)]
        wrong+=torch.count_nonzero(x.view(torch.int16)!=y.view(torch.int16)).item()
        maximum=max(maximum,(x.float()-y.float()).abs().max().item())
    return dict(mismatches=wrong,elements=a.numel(),max_abs=maximum)

def timing(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record();result=fn();b.record();b.synchronize()
    return a.elapsed_time(b)*1000,result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--rows',nargs='+',type=int,default=[512,768,3072,16384])
    ap.add_argument('--rounds',type=int,default=6);args=ap.parse_args()
    assert hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest()==BASE
    torch.set_num_threads(1)
    gen=torch.Generator(device='cuda').manual_seed(args.seed)
    rnd=lambda shape:torch.randn(shape,device='cuda',dtype=torch.bfloat16,generator=gen)
    weight=rnd((8192,40960))/40960**.5
    norm=rnd((8192,))
    lib=ctypes.CDLL(str(ROOT/'research_2026-11-18_lt_audit/rms_linux.so'))
    native=lib.run_bf16_rms;ptr=ctypes.c_void_p
    native.argtypes=[ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_float,ptr];native.restype=ctypes.c_int
    def normalize(x):
        y=torch.empty_like(x)
        assert native(x.data_ptr(),norm.data_ptr(),y.data_ptr(),x.shape[0],8192,1e-5,torch.cuda.current_stream().cuda_stream)==0
        return y
    sys.path.insert(0,str(ROOT/'research_2026-11-18_lt_audit'))
    from real_coupled import configured_rope
    path=ROOT/'sglang/python/sglang/kernels/ops/speculative/fused_kv_materialize.py'
    spec=importlib.util.spec_from_file_location('row_screen_fused_kv',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    h=object.__new__(module.FusedKVMaterializeHelper)
    vals=dict(device=torch.device('cuda'),n_layers=5,flat_kv_weight_t=rnd((8192,1280))/8192**.5,
              layer_out_dim=256,num_kv_heads=1,head_dim=128,rotary_dim=128,k_norm_weights=rnd((5,128)),
              eps_values=torch.full((5,),1e-5,device='cuda'),_workspace_capacity=0,_workspace_dtype=None,
              _proj_workspace=None,_k_workspace=None,_v_workspace=None,_mm_out_supported=True,max_position_hint=131071,
              rotary_emb=S(cos_sin_cache=configured_rope(torch.device('cuda'),500000)),_reserved_rope_cache_len=131072)
    for key,value in vals.items():setattr(h,key,value)
    h._ensure_workspace(max(args.rows),torch.bfloat16)
    report=dict(seed=args.seed,baseline_sha256=BASE,complete=False,cases=[],hardware=torch.cuda.get_device_name(),
                scope='Full40960→8192BF16FC + nativeRMS + five-layerKV/RoPE + packed outputs. Synthetic upstream states and weights; all8rank outputs reconstructed; no TP8 transport.',
                helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    out=HERE/args.out
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    def compute(x,pos,fp32=False):
        fc=torch.mm(x,weight.T,out_dtype=torch.float32).bfloat16() if fp32 else F.linear(x,weight)
        normalized=normalize(fc)
        parts=[]
        h.materialize(normalized,pos,lambda layer,k,v:parts.append(torch.stack((k,v),dim=1)))
        packed=torch.stack(parts,dim=2).contiguous()  # M,2,5,1,128
        return fc,normalized,packed
    for m in args.rows:
        assert m%8==0
        for style in ('random','cancellation'):
            x=rnd((m,40960));positions=torch.arange(m,device='cuda')+1003
            original_weight=weight
            if style=='cancellation':
                # Repeated equal inputs with signs on consecutive K blocks.
                weight=weight.clone();weight[:,1::2]=weight[:,::2]
                x[:,1::2]=-x[:,::2]
                x[:,-1]+=.00390625
            ref=compute(x,positions)
            item=dict(rows=m,style=style,allgather_bytes_per_rank_ring=7*(m//8)*2*5*128*2,
                      gather_allocation_bytes=m*2*5*128*2,comparisons={})
            for variant in ('native','fp32_then_bf16'):
                pieces=[compute(x[s:s+m//8],positions[s:s+m//8],variant!='native') for s in range(0,m,m//8)]
                joined=tuple(torch.cat([v[i] for v in pieces]) for i in range(3))
                item['comparisons'][variant]={stage:compare(a,b) for stage,a,b in zip(('fc','norm','kv'),joined,ref)}
                del pieces,joined
            report['cases'].append(item);save()
            print(m,style,{v:{s:c['mismatches'] for s,c in d.items()} for v,d in item['comparisons'].items()},flush=True)
            if style=='random':
                samples={k:[] for k in ('baseline','control','native_rank','fp32_rank')}
                for _ in range(2):compute(x,positions)
                for turn in range(args.rounds):
                    labels=list(samples)
                    if turn%2:labels.reverse()
                    for label in labels:
                        if label in ('baseline','control'):call=lambda:compute(x,positions)
                        else:call=lambda label=label:compute(x[:m//8],positions[:m//8],label=='fp32_rank')
                        elapsed,output=timing(call);samples[label].append(elapsed);del output
                item['times_us']=samples;item['median_us']={k:statistics.median(v) for k,v in samples.items()}
                save()
            del x,ref,positions
            weight=original_weight
    report['complete']=True;report['peak_mib']=torch.cuda.max_memory_allocated()/2**20;save()

if __name__=='__main__':
    with torch.inference_mode():main()
