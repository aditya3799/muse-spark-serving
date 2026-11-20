"""Verify masked future target features cannot alter an anchor at fixed batch shape."""
import json
from pathlib import Path
import numpy as np
import torch
here=Path(__file__).resolve().parent

def shape_checks(model,drafter):
    rows=[]
    for path in sorted(here.glob("trace_*.npz"))[::3]:
        trace=np.load(path)
        start=int(trace["start"])
        tokens=torch.from_numpy(trace["tokens"][:start+4]).to("cuda")
        out=model(input_ids=tokens[None],output_hidden_states=True,use_cache=False,logits_to_keep=1)
        taps=torch.cat([out.hidden_states[i+1][0] for i in (2,10,18,26)],-1)
        del out
        positions=list(range(start,start+4))
        all_edges=drafter.block_logits(taps,tokens,positions,compare_edges=True)
        joined=all_edges[:3]
        same_ids=(all_edges[1].argmax(-1)==all_edges[3].argmax(-1))
        peak_edge_delta=float((all_edges[1].float()-all_edges[3].float()).abs().max())
        poisoned=taps.clone()
        poisoned[start:]=torch.cos(poisoned[start:].float())*71
        contaminated=tokens.clone()
        contaminated[start+1:]=(contaminated[start+1:] + 197) % 50000
        changed=drafter.block_logits(poisoned,contaminated,positions)
        assert all(torch.equal(x[0],y[0]) for x,y in zip(joined,changed)), "future target leakage at fixed launch shape"
        single=drafter.block_logits(taps[:start],tokens[:start+1],[start])
        li,ls=single[0][0].tolist(),single[1][0].float().cpu().numpy()
        ri,rs=joined[0][0].tolist(),joined[1][0].float().cpu().numpy()
        la,ra=[int(tokens[start])],[int(tokens[start])]
        lp=rp=0
        edge_errors=[]
        shared=[]
        indexed=[]
        for j in range(7):
            common=set(li[j])&set(ri[j])
            shared.append(len(common))
            indexed.append(dict((t,(li[j].index(t),ri[j].index(t))) for t in common))
            prevs=[(0,0)] if j==0 else list(indexed[j-1].values())
            for lpred,rpred in prevs:
                for lc,rc in indexed[j].values():
                    edge_errors.append(abs(float(ls[j,lpred,lc])-float(rs[j,rpred,rc])))
            lp=int(ls[j,lp].argmax());rp=int(rs[j,rp].argmax())
            la.append(li[j][lp]);ra.append(ri[j][rp])
        rows.append(dict(author_vs_pinned_formula_argmax_diffs=int((~same_ids).sum()),author_vs_pinned_peak_delta=peak_edge_delta,trace=path.stem,masked_future_bitwise=True,common_top16_sizes=shared,
                         changed_greedy_slots=sum(x!=y for x,y in zip(la,ra)),
                         aligned_transitions=len(edge_errors),median_aligned_error=float(np.median(edge_errors)),
                         max_aligned_error=max(edge_errors)))
    (here/"future_checks.json").write_text(json.dumps(rows,indent=2))
    print("fixed-shape masked-future checks",rows,flush=True)
