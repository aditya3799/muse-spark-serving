"""Independent symbolic proof of canonical WGMMA packed words feeding native MMA A."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
counts={'A_values':0,'scale_addresses':0,'activation_pairs':0,'epilogue_pairs':0};negatives=0
for bn in (128,256):
 for bk in (64,128):
  nw=bn//64
  for outer in range(3):
   for bank in range(bk//64):
    for it in range(4):
     group=outer*(bk//16)+4*bank+it
     for warp in range(nw):
      for lane in range(32):
       for slab in range(4):
        # Actual LoaderB int4 address, then each of four words.
        word=outer*(bk//16)*(bn*2)+4*(it*(bn//2)+32*warp+lane+nw*32*4*bank)+slab
        for reg in range(4):
         n=warp*64+slab*16+lane//4+(reg%2)*8
         k=group*16+(lane%4)*2+(reg//2)*8
         # Literal canonical-transposed inverse, independent of loader formula.
         for half in range(2):
          kk=k+half;inner=(((n%64)//16*2+(kk//8)%2)*2+(n//8)%2)*2+kk%2
          expect_word=(kk//16)*(bn*2)+(n//64)*128+((n%8)*4+(kk%8)//2)*4+inner//8
          nibble=(inner%8)//2+4*(inner%2)
          assert word==expect_word and nibble==reg+4*half
          counts['A_values']+=1
         # Actual LoaderBS layout1 for WN64:8 contiguousbytes, two per slab.
         sf=group*bn+warp*64+(lane//4)*8+slab*2+reg%2
         oracle=(k//16)*bn+(n//64)*64+(n%8)*8+(n%64)//8
         assert sf==oracle;counts['scale_addresses']+=1
         ordinary_sf=group*bn+warp*64+(lane//4)*8+slab*2+reg//2
         if ordinary_sf!=oracle:negatives+=1
# Activation scalar LDS matches actual cp.async XOR layout with stage phase and K warp offsets.
for bm in (8,16,24,32,48):
 for wm in (8,16,24):
  if bm%wm:continue
  for bk in (64,128):
   for base in range(8):
    for mw in range(bm//wm):
     for bank in range(bk//64):
      for it in range(4):
       for t in range(wm//8):
        for lane in range(32):
         row=mw*wm+8*t+lane//4
         for r in range(2):
          col=bank*64+it*16+(lane%4)*2+r*8
          logical_int4=col//8;pr=row+bm*(logical_int4//8);pc=(logical_int4%8)^((pr+base)%8)
          a0=pr*64+pc*8+col%8;a1=pr*64+pc*8+(col+1)%8
          assert 0<=a0<a1<bm*bk and a1==a0+1
          # inverse reproduces logical token/K pair.
          inv_row=(a0//64)%bm;unxor=(a0%64//8)^((a0//64+base)%8)
          inv_k=(a0//64//bm)*64+unxor*8+a0%8
          assert (inv_row,inv_k)==(row,col);counts['activation_pairs']+=1
# Swap C float2 shfl transposition and orientation-aware writer equal canonical BF16 output swizzle.
for bm in (8,16,24,32,48):
 for wm in (8,16,24):
  if bm%wm:continue
  for bn in (128,256):
   for phase in range(8):
    seen=set()
    for mw in range(bm//wm):
     for nw in range(bn//64):
      for slab in range(4):
       for t in range(wm//8):
        for p in range(2):
         for lane in range(32):
          subrow=(lane%4)*2+(lane%8)//4;row=mw*wm+t*8+subrow
          n=nw*64+slab*16+p*8+(lane//8)*2
          writer=row*32+((2*slab+p)^((subrow+phase)%8))*4+lane//8+nw*(bm*32)
          oracle=(row+bm*(n//64))*32+(((n%64)//8)^((row+phase)%8))*4+(n%8)//2
          assert writer==oracle and writer not in seen;seen.add(writer);counts['epilogue_pairs']+=1
    assert len(seen)==bm*bn//2 and max(seen)<bm*bn//2
assert negatives>0
files=['datatype/dequant.cuh','mma/wgmma.cuh','arith/mainloop_arith.cuh','memory/s2r_loader/loader_b.cuh','memory/s2r_loader/loader_bs.cuh','memory/g2s_loader/loader_a.cuh','epilogue/smem_writer.cuh','epilogue/smem_reducer.cuh']
r={'status':'CPU_MAPPING_PASS_NOT_PIPELINE_GPU_PROOF','counts':counts,'wrong_standard_scale_selector_cases':negatives,'sources':{f:hashlib.sha256((V/f).read_bytes()).hexdigest()for f in files},'conclusion':'Canonical WGMMA packed word can feed standard native MMA A directly: decoded register order [N0K0,N8K0,N0K8,N8K8]. Both packed-word remap and BF16word permutation are unnecessary when scale selector is changed to physical A row selector reg%2.','exact_required_scale_edit':'Within may_apply_bs_and_zp_on_b, nested i,k indexes reg=i*2+k. For swap direct A use bs_f16_ptr[k], not bs_f16_ptr[i]. Leave exponent correction, scale conversion, BF16 multiplications and global Ctx::kUseWgmma untouched.','scope':'WN64, BN128/256, BK64/128, WK64, GS16, BF16 activations/FP32accum/NVFP4+E4M3; noZP/bias/inputscale/streamK/TMA/WS/packedK. Distinct symbolic N and K16 scaleaddresses used, so a repeated scale cannot hide wrong selection.','epilogue':'Proof covers transposed writer after shfl with num_write_splits1; split2 deliberately not admitted. K reduction must remain FP32 before writer BF16 cast.'}
(P/'SWAP_DIRECT_A_MAPPING_PROOF.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'counts':counts,'wrong_scale_negative':negatives}))
