"""Run original and earliest-tap forwards for Muse's native 62-layer ordering on actual BF16 arrays."""
from pathlib import Path
from types import SimpleNamespace as NS
import json,sys,torch
H=Path(__file__).resolve().parent;D=H/"decode_overlap"
sys.path.insert(0,str(H.parent/"research_2026-11-18_rl"))
import bench_aux as ref

def main():
    torch.set_num_threads(1)
    parent=ref.forward((ref.PARENT/"candidate_sources"/ref.MODEL).read_text());candidate=ref.forward((D/"tap_muse_spark.py").read_text())
    pack=ref.pack_function();candidate.__globals__["pack_aux_hidden_states"]=pack
    count=0;entries=0;fixture=[]
    for seed in (53107,35771):
        gen=torch.Generator().manual_seed(seed)
        for m in (1,7,32):
            for active in (True,False):
                for sp in (True,False):
                    for taps in ([],[1,16,30,44,59],[0],[61],[1,3,3],[5,2,0]):
                        for fn in (parent,candidate):
                            fn.__globals__["layernorm_sp"]=NS(runs_sp=lambda _:sp,sp_entry_scatter=lambda x:x)
                        xs=[torch.randn(m,64,generator=gen) for _ in range(62)]
                        visited=[]
                        class Layer:
                            def __init__(self,i):self.i=i
                            def self_attn_input_norm(self,s):return s
                            def __call__(self,pos,stream,inp,batch):visited.append(self.i);return xs[self.i],inp
                        model=NS(embed_norm=None,residual_dtype=torch.float32,operator_dtype=torch.bfloat16,
                                 layers_to_capture=taps,layers=list(map(Layer,range(62))),norm=lambda x:x)
                        embedding=xs[0].bfloat16()
                        a=ref.call(parent,model,NS(forward_mode=1),embedding,pack)
                        visited.clear();when=[]
                        mock=NS(record=lambda:when.append(visited[-1]))
                        batch=NS(forward_mode=1,dflash_context_ready_event=mock if active else None)
                        b=ref.call(candidate,model,batch,embedding,pack)
                        assert len(a)==len(b) and all(torch.equal(x,y) for x,y in zip(a,b))
                        eligible=active and not sp and bool(taps) and len(set(taps))==len(taps)
                        assert when==([max(taps)] if eligible else [])
                        if eligible:assert b[1] is batch.dflash_tapped_context
                        entries+=sum(x.numel() for x in a);count+=1
                        if when==[59]:fixture.append((when[0],visited[-2:]))
    r=dict(cases=count,compared_values=entries,official_taps=[1,16,30,44,59],remaining_after_early_signal=[60,61],
           official_marker_positions=[w for w,_ in fixture],sp_and_missing_fallback=True,
           scope="Native scored Muse forward AST and pinned pack implementation on synthetic 62-layer FP32 residuals and BF16 captures. Fake event records its position; timing and CUDA event behavior excluded.")
    (D/"tap_cpu.json").write_text(json.dumps(r,indent=2));print(r)
if __name__=="__main__":main()
