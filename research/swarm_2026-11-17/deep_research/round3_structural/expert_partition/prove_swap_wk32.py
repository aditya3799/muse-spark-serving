"""Independent BK64/WK32 source address and exact K-bank sequence proof."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
counts={'words':0,'scales':0,'activation_pairs':0,'K_terms':0};records=[]
for K in (128,4096):
 banks=[[outer+32*b+16*i+kk for outer in range(0,K,64) for i in range(2) for kk in range(16)]for b in range(2)]
 assert sorted(banks[0]+banks[1])==list(range(K))
 assert banks[0][:64]==list(range(32))+list(range(64,96))
 assert banks[1][:64]==list(range(32,64))+list(range(96,128))
 counts['K_terms']+=K;records.append({'K':K,'bank0_first64':banks[0][:64],'bank1_first64':banks[1][:64],'final_tree':'RN32(bank0+bank1)'})
for bn in (128,256):
 nw=bn//64
 for outer in range(3):
  for bank in range(2):
   for it in range(2):
    group=outer*4+bank*2+it
    for warp in range(nw):
     for lane in range(32):
      for slab in range(4):
       actual=outer*4*bn*2+4*(it*(bn//2)+32*warp+lane+nw*32*2*bank)+slab
       for reg in range(4):
        n=warp*64+slab*16+lane//4+(reg%2)*8;k=group*16+(lane%4)*2+(reg//2)*8
        for half in range(2):
         kk=k+half;inner=(((n%64)//16*2+(kk//8)%2)*2+(n//8)%2)*2+kk%2
         expected=(kk//16)*bn*2+(n//64)*128+((n%8)*4+(kk%8)//2)*4+inner//8
         assert actual==expected and (inner%8)//2+4*(inner%2)==reg+4*half;counts['words']+=1
        sf=group*bn+warp*64+(lane//4)*8+slab*2+reg%2
        assert sf==(k//16)*bn+(n//64)*64+(n%8)*8+(n%64)//8;counts['scales']+=1
for bm in (8,16,24):
 for phase in range(8):
  for bank in range(2):
   for it in range(2):
    for t in range(bm//8):
     for lane in range(32):
      for r in range(2):
       row=t*8+lane//4;column_int4=it*2+32*bank//8+r
       physicalrow=bm*(column_int4//8)+row;physicalcol=(column_int4%8)^((physicalrow+phase)%8)
       addr=physicalrow*64+physicalcol*8+(lane%4)*2
       invrow=(addr//64)%bm;invcol=(addr//64//bm)*64+((addr%64//8)^((addr//64+phase)%8))*8+addr%8
       assert(invrow,invcol)==(row,bank*32+it*16+r*8+(lane%4)*2)
       assert 0<=addr<addr+1<bm*64;counts['activation_pairs']+=1
files=['kernel/humming.cuh','memory/s2r_pipeline.cuh','memory/s2r_loader/loader_b.cuh','memory/s2r_loader/loader_bs.cuh','epilogue/smem_reducer.cuh']
r={'status':'CPU_WK32_MAPPING_AND_K_SEQUENCE_PASS','counts':counts,'banks':records,'sources':{f:hashlib.sha256((V/f).read_bytes()).hexdigest()for f in files},'pipeline_requirement':'Keep originalkernelwhilebodybyteidentical: kWarpIters=2, producer-refill predicate warp_iter_id==0, samecpasyncstage3consumerwait/refill schedule. MMA sync introduces no pendingtensorcore operand lifetime.','arithmetic_contract':'Two32wideKbanks, not64wide. Bank0 visitsK0..31,64..95,...; bank1K32..63,96..127,...; both retain original16wideMMA callorder and finalFP32bank0+bank1 beforeBF16.','metadata':'WN64/GS16B4FP8canonicalword andscale layout unchanged; directAselectorreg%2 remains mandatory.','limits':'CPU proof only, BK64/WK32 needs separate nativeFP32microprobe andcpasyncpipeline gates. N4096/E8fullhelper remainsheld by currentrenderer admission.'}
(P/'SWAP_WK32_INDEPENDENT_MAPPING_PROOF.json').write_text(json.dumps(r,indent=2)+'\n');print(counts)
