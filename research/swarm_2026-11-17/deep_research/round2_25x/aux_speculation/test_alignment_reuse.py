"""CPU mathematical alignment contract only; not an implementation or GPU test."""
import itertools,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3]
def align(counts,block):
 rows=[];experts=[];route=0
 for e,c in enumerate(counts):
  rows.extend((e,v) for v in range(route,route+c));route+=c
  rows.extend([None]*((-c)%block));experts.extend([e]*((c+block-1)//block))
 return rows,experts
cases=0
for counts in itertools.product([0]+list(range(17,33)),repeat=3):
 rows16,experts16=align(counts,16);rows32,experts32=align(counts,32)
 assert rows16==rows32
 assert len(rows16)%32==0
 assert experts16[::2]==experts32
 for i in range(len(experts32)):
  assert experts16[2*i]==experts16[2*i+1]
  assert all(v is None or v[0]==experts16[2*i] for v in rows16[32*i:32*i+32])
 cases+=1
negative=[]
for counts in ([33,17],[17,33,17],[1,17],[17,1,17]):
 a,ae=align(counts,16);b,be=align(counts,32)
 assert a!=b or ae[::2]!=be
 mixed=[]
 for i in range((len(a)+31)//32):
  ids=set(v[0] for v in a[32*i:32*i+32] if v is not None)
  if len(ids)>1:mixed.append(i)
 negative.append({'counts':counts,'padded16':len(a),'padded32':len(b),'mixed_expert_tiles_when_reused':mixed,'rejected':True})
files=[ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming/scheduler.cuh',ROOT/'sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py',ROOT/'sglang/python/sglang/kernels/aot/csrc/moe/moe_align_kernel.cu',ROOT/'sglang/python/sglang/kernels/ops/moe/muse_spark_routing.py',ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py']
r={'status':'CONDITIONAL_CPU_PROOF_PASS_NO_IMPLEMENTATION','positive_count_vectors':cases,'negative_controls':negative,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'scope':'Canonical mathematical stable order model; actual GPU align uses parallel route placement. Reuse preserves the already-produced BM16 order, so no assumption that two separate align launches have identical within-expert order is required.'}
(P/'ALIGNMENT_REUSE_CPU.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'positive_cases':cases,'negative_cases':len(negative)}))
