"""CPU address/lifetime model for proposed compact grouped fusedW13 two-buffer path."""
import hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[3];H=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
checks=0;cases=[]
for bm in [8,16,24,32,48,64,96,128,160,176]:
 for bn in [128,256]:
  for splits in [1,2]:
   for offset,count in [(0,0),(0,1),(3,bm-1),(7,bm),(11,bm+1),(13,2*bm+3)]:
    nraw=8192;iwidth=nraw//2;end=offset+count;capacity=end+19;seen=set()
    for tile in range((count+bm-1)//bm):
     row_offset=offset+tile*bm;live=min(max(end-row_offset,0),bm)
     for nb in [0,1,nraw//bn-1]:
      for split in range(splits):
       for smem_offset in range(bm*bn//8//splits):
        smem_row,smem_col=divmod(smem_offset,8)
        row=smem_row%(bm//splits)+(bm//2*split if splits==2 else 0)
        col=smem_row//(bm//splits)*8+smem_col
        if row>=live:continue
        ptr_uint2=nb*(bn*2//16)+min(row_offset,end)*(nraw//8)
        addr_uint2=ptr_uint2+row*(nraw//8)+col
        logical_bf16=addr_uint2*4
        assert logical_bf16==(row_offset+row)*iwidth+nb*(bn//2)+col*4
        assert 0<=logical_bf16 and logical_bf16+4<=capacity*iwidth
        key=(row_offset+row,nb,col);assert key not in seen;seen.add(key);checks+=1
    assert len(seen)==count*3*(bn//8)
    cases.append({'bm':bm,'bn':bn,'write_splits':splits,'absolute_expert_begin':offset,'count':count,'live_stores':len(seen)})
# Symbolic opaque-value pipeline checks separate read and write lifetimes at whole-layer boundaries.
rng=random.Random(31);liveness=0
for slots in [8,64,1024]:
 for generation in range(4):
  local=[s for s in range(slots) if generation!=3 and rng.random()<.4];rng.shuffle(local)
  b0=[None]*slots;b1=[None]*slots
  for j,slot in enumerate(local):b0[j]=('X',slot)
  # Both routing partitions finish before any B0 down writes.
  for j,slot in enumerate(local):assert b0[j]==('X',slot);b1[j]=('ACT',slot)
  for j,slot in enumerate(local):assert b1[j]==('ACT',slot);b0[j]=('DOWN',slot)
  # All W2 consumers finish before B1 scatter reuses activation storage.
  for j,slot in enumerate(local):b1[slot]=b0[j]
  for slot in local:assert b1[slot]==('DOWN',slot);liveness+=1
# Adversarial compact permutation proves that scatter while a W2 reader remains live can overwrite it.
b1=[('ACT',1),('ACT',0)];b0=[('DOWN',1),None];b1[1]=b0[0];assert b1[1]!=('ACT',0)
report={'status':'CPU_PROPOSED_COMPACT_FUSED_BUFFER_CONTRACT_PASS','scope':'Independent legacyC output-address/lifetime model; renderer/native code review still required. No TMA C or numerical proof.','source_hashes':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [H/'epilogue/gmem_writer.cuh',H/'scheduler.cuh',ROOT/'swarm_2026-11-17/deep_research/coordination/w13_fused_epilogue.py']},'address_checks':checks,'cases':cases,'opaque_lifetime_checks':liveness,'lifetime_negative':'Early B1 scatter can overwrite a still-live compact activation row; whole-W2 completion is required.','admission':['Both selected W13 light and heavy kernels must have fused half-width writer before allocating2*S*H BF16. Empty current partition does not permit an unfused fallback to be ignored.','Nraw=2H, activationH, downH; H=I. Interleaved W13 weights and scales must match every admitted branch.','TMA C false: rawN scheduler remains, output is half width via8-byte uint2 stores replacing16-byte int4 stores. No output descriptor exists in this scope.','Ordinary Humming C shape check expectsrawN; explicit fused launcher checks halfwidth and exact allocation. Mixed TMA A/B/BS launcher must pass descriptor-by-value, not all rawpointers.','B0[S,H]: gatherX then compactdown after ALL W13 readers complete. B1[S,H]: fused activation then canonical scatter after ALL W2 readers complete.','Finalizer consumes canonical slots in original0..7 order and skips invalidroutes; staleinactive bytes are not expected outputs.','Graph replay rebuilds compact maps/offsets and preserves pointer capacities; allremote, last expert, changing cutoff require numerical gates.','Shared reduce storage still holds rawW13 tile; global2SH does not reduce kernel shared memory requirement.'],'deferred_TMA_C_requirements':['Half globalN and globalrowstride, half output tile geometry/coordinate, fusedsharedlayout/storegroupcompletion all require separate descriptor-aware implementation. Retaining originalTMA C descriptor cannot implementhalfwidth safely.'],'recommendation':'First compose with TMA A/B/BS and legacyC only; reject unsupportedmetadata beforebufferallocation. Keep original3SH fallback outside the2SH helper.'}
(P/'COMPACT_FUSED_TWO_BUFFER_INDEPENDENT_MODEL.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'address_checks':checks,'cases':len(cases),'lifetime_checks':liveness}))
