from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming';C=P.parent/'hopper_moe/chord_reference'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=0; skipped=0
for width in (8,16):
 for tiles in range(1,13):
  capacity=width*tiles
  for live in range(capacity+1):
   slots=list(range(live))+[2**31-1]*(capacity-live)
   enabled=[j==0 or slots[j*width]<capacity for j in range(tiles)]
   for row in range(live):
    assert enabled[row//width];checks+=1
   for j,on in enumerate(enabled):
    if not on:assert all(v>=capacity for v in slots[j*width:(j+1)*width]);skipped+=1
# First-slot test cannot be used on arbitrary holey metadata.
holey=[0]*8+[2**31-1,1]+[2**31-1]*6
assert holey[8]>=16 and holey[9]<16
wgmma=(V/'mma/wgmma.cuh').read_text();wmma=(C/'chord_kernels/operator/include/humming/mma/wmma.cuh').read_text()
assert 'static_assert(WarpShape::M == MmaShape::M)' in wgmma
assert 'regs_c[0][delta_j + j][0]' in wgmma
assert 'if (j != 0 && j >= jmax) continue;' in wmma
r={'status':'CPU_SOURCE_FALSIFICATION_OF_DIRECT_WGMMA_TRANSFER','sources':{str(p):sha(p) for p in [C/'LICENSE',C/'docs/design.md',C/'chord_kernels/operator/include/humming/mma/wmma.cuh',C/'chord_kernels/operator/include/humming/kernel/humming.cuh',V/'mma/wgmma.cuh',V/'mma/wmma.cuh',V/'scheduler.cuh']},'license':'Chord Apache-2.0, Humming-derived notices retained. No Chord CUDA code copied in this study.','checks':{'retained_live_rows':checks,'proven_empty_skipped_tiles':skipped,'negative_holey_metadata':True},'contract':['Valid sorted route slots form a contiguous prefix within each aligned expert block; padding sentinel is >= output rows (shape_m*top_k).','Sorted metadata is complete and visible before checking shared wr_row_index; derive tile eligibility anew on every dynamic graph replay.','Each valid original slot has unique destination ownership. Invalid rows are masked in original epilogue and never affect valid-row reductions.','j=0 unconditional is safe even for an all-invalid local warp tile, but produces no useful work; a launched expert tile normally has at least one route.','A sentinel at the first row only proves the whole subtile invalid under prefix packing; arbitrary holey input violates the guard.','All K steps, register-bank association, FP32 accumulation, BF16 conversion, scale and final reduction order for retained rows remain unchanged.'],'finding':'Chord swap-AB WMMA j spans token-N8 subtiles. Pinned WGMMA uses one native token-width instruction (WarpShape.M==MmaShape.M); its j spans output-channel fragments. Guarding that j by token occupancy drops valid output columns. No safe direct j-guard candidate exists for this WGMMA geometry.','distinction_from_prior':'M2 changes tile geometry and weight reuse; compact changes route storage/addressing. Semi-static pruning leaves geometry unchanged and omits only entirely padded token sub-instructions. It would be new in a swap-AB implementation, but has no independent sub-instruction on pinned WGMMA.','ordinary_wmma_alternative':'Pinned ordinary WMMA m spans 16-token subtiles and can be guarded using wr_row_index[ctx.m_warp_offset()+16*m]. Small-M WM16 has only m0 and yields no savings. Large WM48 is a concrete possible case, but parent deferred implementation until higher priority reviews.','decision':'No compiler lease requested, no naive WGMMA guard, no no-op small-M variant, no numerical/performance claim.'}
(P/'SEMISTATIC_TOKEN_TILE_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r['checks']))
