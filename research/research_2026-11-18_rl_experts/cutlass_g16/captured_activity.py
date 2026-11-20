"""Distinct resident experts in real public GLM route captures for fallback traffic sensitivity."""
import json
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent;root=here.parents[1];place=root/"research_2026-11-18_rl/placement"
models={}
for name in ("glm52","glm53"):
    src=json.loads((place/(name+"_source.json")).read_text())
    data=np.load(place/(name+"_routes.npz"))["arr"]
    if data.ndim!=4 or data.shape[-1]!=8 or int(data.max())>=256:
        continue
    rows=[]
    for m in (8,32,128,256,768):
        windows=[];busiest=[]
        for chunk in data:
            w=chunk.shape[0]//m
            for layer in range(chunk.shape[1]):
                ids=chunk[:w*m,layer,:].astype(np.int32).reshape(w,m*8)
                bins=ids+256*np.arange(w,dtype=np.int32)[:,None]
                hist=np.bincount(bins.ravel(),minlength=w*256).reshape(w,8,32)
                distinct=(hist>0).sum(axis=-1)
                windows.extend(distinct.ravel())
                chosen=hist.sum(axis=-1).argmax(axis=-1)
                busiest.extend(distinct[np.arange(w),chosen])
        a=np.array(windows);b=np.array(busiest)
        rows.append(dict(m=m,all_rank_windows=len(a),p05_p50_p95_distinct=np.quantile(a,[.05,.5,.95]).tolist(),
                         mean_copy_MiB=float(a.mean()*54),max_copy_MiB=int(a.max()*54),
                         busiest_rank_median_copy_MiB=float(np.median(b)*54)))
    models[name]=dict(dataset=src['repo'],revision=src.get('revision'),capture=src.get('capture'),
                      shape=data.shape,summary=rows)
(root/"research_2026-11-18_rl_experts/cutlass_g16/captured_activity.json").write_text(json.dumps(models,indent=2))
for name,r in models.items():print(name,[(a['m'],a['p05_p50_p95_distinct'],round(a['mean_copy_MiB'])) for a in r['summary']])
