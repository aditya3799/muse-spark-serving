import json,re
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
path=ROOT/'references/gauravjain14-muse-okc-submissions/muse_attention_decode/01_fused_flat_splitkv_fp8/evidence/judge_before_calib_sglang_e9c98045.txt'
text=path.read_text(encoding='utf-8',errors='replace')
cases={}
for p in re.split(r'(?m)^profile \d+: ',text)[1:]:
    title=p.splitlines()[0];batch=int(re.search('batch=(\\d+)',title)[1]);layer=re.search('layer=(\\w+)',title)[1];mode=re.search('load=(\\w+)',title)[1]
    items=[]
    for row in p.splitlines()[2:]:
        m=re.search(r'^(.*?)\s{2,}([0-9]+\.[0-9]+)\s',row)
        if m:items.append((m[1],float(m[2])))
    total=next(v for k,v in items if k=='total')
    changed={k:v for k,v in items if any(s in k for s in ('_qk_rms_norm','store_kvcache','elementwise_kernel','copy_kernel'))}
    # Named elementwise rows are truncated in organizer tables; aggregate as
    # potentially entry-related and retain names to expose the ambiguity.
    cases[f'{mode}_{batch}_{layer}']=dict(total_gpu_us=total,entry_related_gpu_us=sum(changed.values()),kernel_rows=changed)
summary={}
for mode in ('decode','verify'):
 for b in (1,16,32):
    s=cases[f'{mode}_{b}_sliding'];f=cases[f'{mode}_{b}_full']
    a=46*s['entry_related_gpu_us']+16*f['entry_related_gpu_us'];total=46*s['total_gpu_us']+16*f['total_gpu_us']
    summary[f'{mode}_{b}']=dict(entry_related_us_per_62_layers=a,attention_track_gpu_us_per_62_layers=total,entry_fraction=a/total)
out=dict(job='e9c98045b4184779',scope='Independent H200 attention microbenchmark; four times truncated kernel labels; no serving projection. Ratios use sums of per-kernel GPU times, which may include PDL overlap.',cases=cases,summary=summary)
(H/'donor_h200.json').write_text(json.dumps(out,indent=2));print(summary)

