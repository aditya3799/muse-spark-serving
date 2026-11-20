"""CPU-only canonical-slot and bounded-scratch research; no CUDA imports."""
import collections, hashlib, json, pathlib, random
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[3]
OLD=ROOT/'swarm_2026-11-17/deep_research/expert_partition'
COORD=OLD.parent/'coordination'
def padded(counts, choices):
 return sum(((n+choices[e]-1)//choices[e])*choices[e] for e,n in counts.items())
def main():
 rows=[];checks=0
 for m in (8,128,576,768,1024,1536,3072):
  for mode in ('uniform','hot_rank','mixed','all_remote','tail_invalid'):
   rng=random.Random(481+m);global_ids=[]
   for t in range(m):
    pool=range(32) if mode=='hot_rank' or (mode=='mixed' and t<m//2) else range(256)
    routes=rng.sample(pool,8)
    if mode=='all_remote':routes=rng.sample(range(32,256),8)
    if mode=='tail_invalid' and t>=m//2:routes=[-1]*8
    global_ids.append(routes)
   flat=[e if 0<=e<32 else -1 for ids in global_ids for e in ids]
   counts=collections.Counter(e for e in flat if e>=0)
   for bm,heavy in ((32,128),(48,176),(64,160)):
    choices={e:bm if n<=bm else heavy for e,n in counts.items()}
    light=[s for s,e in enumerate(flat) if e>=0 and counts[e]<=bm]
    dark=[s for s,e in enumerate(flat) if e>=0 and counts[e]>bm]
    active=[s for s,e in enumerate(flat) if e>=0]
    assert set(light).isdisjoint(dark) and sorted(light+dark)==active
    # A compact position requires an explicit reverse map; final traversal stays k=0..7.
    reverse={s:c for c,s in enumerate(active)}
    for t in range(m):
     before=[s for s in range(t*8,(t+1)*8) if flat[s]>=0]
     after=[active[reverse[s]] for s in range(t*8,(t+1)*8) if s in reverse]
     assert before==after
    chunk_padded=0
    for start in range(0,m,256):
     chunk_counts=collections.Counter(e for e in flat[start*8:min(m,start+256)*8] if e>=0)
     chunk_padded+=padded(chunk_counts,choices)
    full_padded=padded(counts,choices)
    rows.append(dict(m=m,pattern=mode,light_m=bm,heavy_m=heavy,local_routes=len(active),route_capacity=m*8,baseline_bytes=m*8*3*4096*2,composed_bytes=m*8*2*4096*2,ideal_compact_bytes=len(active)*2*4096*2,full_padded_rows=full_padded,token_chunk256_padded_rows=chunk_padded,chunk_padding_ratio=chunk_padded/full_padded if full_padded else None))
    checks+=2+m
 sources=[OLD/'scalable_partition.py',COORD/'w13_fused_epilogue.py',COORD/'probe_w13_fusion_native_e32.py']
 report=dict(status='CPU_CONTRACT_PASS',checks=checks,cases=len(rows),sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},rows=rows,contract=[
 'All local routes execute exactly once; light/heavy classification is exhaustive and disjoint.',
 'Canonical row s=token*topk+slot remains unchanged in both GEMMs and finalizer.',
 'Interleave raw W13 gate/up weight AND group-scale rows before canonical packing; global expert scale unchanged.',
 'For large prefill preserve baseline block-K and warp-K in light configs. FP32 accumulation, no StreamK.',
 'Fused W13 preserves GEMM BF16 round, explicit non-FTZ scale multiply, BF16 round, stock SiLU/multiply, BF16 round.',
 'Both W13 partitions complete before W2. Activation and down buffers do not alias.',
 'W2 configuration, BF16 postscale boundary, final norm, weighting and slot-order sum are unchanged.',
 'Graph allocation uses worst-case static R=M*topk capacity. Expected local count is not a safe capacity bound.',
 'Remote slots need not be initialized only when finalizer predicates before loads; changed graph IDs must regenerate classification and alignment.'
 ],limits=['CPU ownership proof only; no arithmetic or performance validation','Tile pairs include illustrative SM86 and SM90 regimes, not full selector enumeration','Ideal compact bytes are an unattained lower bound; no safe graph-capacity implementation','Token chunking preserves full-batch choices in this model yet increases padding; weight rereads unmodeled','No measured serving speedup'])
 (P/'CPU_CONTRACT.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'checks':checks,'cases':len(rows),'uniform768':[r for r in rows if r['m']==768 and r['pattern']=='uniform']}))
if __name__=='__main__':main()
