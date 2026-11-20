"""Generic configured-shape admission and phase costs; source-only, no torch."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
LIMIT=1_800_000_000  # 200MB remaining under decimal2GB for unmodeled overhead.
MARGIN=32<<20


def admit_rows(configured,*,k,n,tp,rank_workspace_bytes,rank_other_bytes,limit=LIMIT):
    """One CPU-collectively declared configuration; all-rank worst cost wins.

    rank_other_bytes must include retained unrelated feature state, graph-private
    storage, and any already-retained N workspace. No cudaMemGetInfo inference can
    distinguish baseline weights/pools from candidate allocations automatically.
    """
    if tp not in (2,4,8) or len(rank_workspace_bytes)!=tp or len(rank_other_bytes)!=tp:
        return dict(accepted=[],reason='inconsistent rank manifest')
    if any(v<0 for v in rank_workspace_bytes+rank_other_bytes):
        return dict(accepted=[],reason='negative reservation')
    costs=[]
    for m in sorted(set(configured)):
        if not isinstance(m,int) or m<512 or m%tp:continue
        per_rank=[m*(2*k+2*n+2*n//tp)+3*w+MARGIN+other
                  for w,other in zip(rank_workspace_bytes,rank_other_bytes)]
        costs.append(dict(rows=m,rank_bounds=per_rank,accepted=max(per_rank)<limit))
    return dict(accepted=[c['rows'] for c in costs if c['accepted']],costs=costs,
                limit=limit,scope='Declared source bounds, not allocator measurements')

W=8_519_680
normal=1_321_355_776
wgs=61_865_984
kv=128<<20
configured=[512,1024,2048,4096,8192,16384]
scenarios={name:admit_rows(configured,k=40960,n=8192,tp=8,
        rank_workspace_bytes=[W]*8,rank_other_bytes=[other]*8)
    for name,other in [('early_no_arenas',0),('post_normal',normal),
                       ('post_normal_with_wgs_graph_storage',normal+wgs),
                       ('post_normal_wgs_and_kv',normal+wgs+kv),
                       ('recapture_all_resident_plus_N_workspace',normal+wgs+kv+W)]}
# One rank with larger native workspace must constrain EVERY rank identically.
scenarios['asymmetric_workspace']=admit_rows(configured,k=40960,n=8192,tp=8,
    rank_workspace_bytes=[W]*7+[256<<20],rank_other_bytes=[0]*8)
# This plan is built only from the configured list. It never invents max-fit rows.
assert scenarios['early_no_arenas']['accepted']==configured
assert scenarios['post_normal_wgs_and_kv']['accepted']==[512,1024,2048]
assert 16384 not in scenarios['asymmetric_workspace']['accepted']
assert not admit_rows(configured,k=40960,n=8192,tp=8,rank_workspace_bytes=[W],rank_other_bytes=[0]*8)['accepted']

normal_cal=1_749_451_072
rows16k=16384*(2*40960+2*8192+2*8192//8)+3*W+MARGIN
phase=[
 dict(phase='early row calibration',bytes=rows16k,conditions='No normal Arena, no WGS graph storage, no KV staging'),
 dict(phase='N calibration after row',bytes=(128<<20)+W,conditions='N own128MiB bound plus retained row workspace'),
 dict(phase='normal qualification',bytes=normal_cal+2*W,conditions='Before decode capture; no WGS retained graph storage and KV staging unallocated'),
 dict(phase='normal qualification after prior WGS capture',bytes=normal_cal+2*W+wgs,conditions='Not below1.8GB planning bound; require admission/fallback, no unconditional late qualification'),
 dict(phase='normal plus row/N resident',bytes=normal+2*W,conditions='One serving stream, native handles not counted'),
 dict(phase='normal+row/N+conservative WGS capture delta',bytes=normal+2*W+wgs,conditions='WGS temporary tile storage may be graph-retained; exact pool delta must be measured, do not multiply by layers'),
 dict(phase='future optional KV adds128MiB',bytes=normal+2*W+wgs+kv,conditions='KV currentlydisabled; FP8 model gate unresolved'),
 dict(phase='all-feature row16K fastpath',bytes=normal+2*W+wgs+kv+16384*8192*2//8,conditions='Local output extra; gathered full output replaces baseline native output'),
 dict(phase='all-feature row16K fallback conservative',bytes=normal+2*W+wgs+kv+16384*8192*2+16384*8192*2//8,conditions='Native full and gathered full coexist; local contiguous view ordinarily aliases native, conservatively counted'),
]
for p in phase:p['below_planning_limit']=p['bytes']<LIMIT
out=dict(scope='Source phase/admission proposal. No combined allocation attempted and no serving memory claim.',
  baseline_exclusions=['Original FC/aux and target weights','Stock KV pools','Stock graph baseline footprint','Existing live FC input and one final FC output'],
  key_correction='WGS61.9MB is per-call tiles, not persistent facade state; graph capture can retain storage. Prefill captures before decode, so early normal qualification can precede this lifetime.',
  phase=phase,scenarios=scenarios,
  deployment_rules=['CPU collectively declare geometry, workspace size and ownership reservations before qualification; use all-rank worst bound',
    'Retain exact registered shapes only; no qualification extrapolation',
    'No new qualification after capture unless current phase reservation fits',
    'Abort startup on fatal CUDA or collective failure; do not retry a one-shot half-initialized worker',
    'One-shot registration is not a weight-update or stream-reconfiguration lifecycle hook',
    'Measure current allocated/reserved and native memory without erasing peak history; actual graph-private and NCCL deltas remain required'])
(HERE/'early_projection_unapplied/phase_admission.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(phase=phase,accepted={k:v['accepted'] for k,v in scenarios.items()}),indent=2))
