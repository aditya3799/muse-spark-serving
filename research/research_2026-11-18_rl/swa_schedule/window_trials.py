"""Drive the frozen native 62-layer stack with independently rebased sliding K/V metadata."""
import hashlib
import sys
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/"native_stack.py").read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,old
    source=source.replace(old,new)

replace("from snapshot import snapshot","from snapshot import snapshot\nfrom compact_window import prepare")
replace('default=["schedule","direct_schedule","private_schedule"]','default=["schedule","align128","align1920"]')
replace('choices=["schedule","direct","private","direct_schedule","private_schedule"]','choices=["schedule","align128","align1920"]')
replace('("native_smoke_" if args.smoke else "native_stack_")','("window_smoke_" if args.smoke else "window_stack_")')
replace('def allocate():','def allocate(name):\n            align=int(name[5:]) if name.startswith("align") else 0\n            width=min(cap,left+n+align-1) if align else cap')
replace('sw=torch.zeros((b,cap)', 'short=seq.clone(),short_cu=cuq.clone(),align=align,sw=torch.zeros((b,width)')
replace('name:allocate()','name:allocate(name)')
replace('def plan(ll,window):','def plan(ll,window,width=cap):')
replace('max_seqlen_k=cap,','max_seqlen_k=width,')
replace('if direct:fast','if st["align"]:\n                prepare(*args0,short=st["short"],short_cu=st["short_cu"],swa_suffix_tokens=left+n,alignment=st["align"])\n            elif direct:fast')
replace('if reuse:\n                slide=plan(l,(left,0))','slen=st["short"] if st["align"] else l\n            scu=st["short_cu"] if st["align"] else cu\n            if reuse or st["align"]:\n                slide=plan(slen,(left,0),st["sw"].shape[1])')
replace('cache_seqlens=l,cache_batch_idx','cache_seqlens=(l if full else slen),cache_batch_idx')
replace('cu_seqlens_k_new=cu,','cu_seqlens_k_new=(cu if full else scu),')
replace('out=dict(scope="','out=dict(swa_rebase_implementation=hashlib.sha256((here/"compact_window.py").read_bytes()).hexdigest(),scope="')
replace('MatchedImpossible','') if False else None
replace('ref=states["baseline"]["out"]','''ref=states["baseline"]["out"]
            for arm in names:
                state=states[arm]
                if not state["align"]:continue
                absolute=states["baseline"]["l"]
                base=((absolute-left-n).clamp_min(0)//state["align"])*state["align"]
                lens=absolute-base
                assert torch.equal(state["short"],lens)
                assert torch.equal(state["short_cu"][1:],lens.cumsum(0,dtype=torch.int32))
                col=torch.arange(state["sw"].shape[1],device=dev)[None,:].expand(b,-1)
                # Check every reachable read for every query, including both inclusive ends.
                visible=(col<lens[:,None]) & (col>=(lens[:,None]-left-n).clamp_min(0))
                rr=torch.arange(b,device=dev)[:,None].expand_as(col)
                assert torch.equal(state["sw"][visible],states["baseline"]["sw"][rr[visible],(col+base[:,None])[visible].long()])''')
exec(compile(source,str(here/"native_stack.py"),"exec"),{"__file__":str(here/"native_stack.py"),"__name__":"__main__"})
