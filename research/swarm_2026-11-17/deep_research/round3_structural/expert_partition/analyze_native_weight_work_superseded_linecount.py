import collections,hashlib,json,re,struct,random
from pathlib import Path
P=Path(__file__).resolve().parent;D=P.parents[1];ROOT=P.parents[3];H=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
paths=list((P.parent/'hopper_moe/offline_compact_01').glob('*/kernel.sass'))+[D/'round2_25x/hopper_moe/offline_wait1_01/wait1/w13_m32_s4_control/kernel.sass',D/'round2_25x/hopper_moe/offline_wait1_01/wait1/w13_m8_s4_control/kernel.sass']
static=[]
for path in paths:
 lines=[x for x in path.read_text().splitlines() if re.search(r'/\*[0-9a-f]+\*/',x)];hs=[i for i,x in enumerate(lines) if ' HGMMA.' in x]
 windows=[]
 for j in range(0,len(hs)-2,2):
  part=lines[hs[j]:hs[j+2]]
  windows.append({'begin':re.search(r'/\*(.*?)\*/',part[0]).group(1),'end_exclusive':re.search(r'/\*(.*?)\*/',lines[hs[j+2]]).group(1),'HMUL2_BF16':sum(' HMUL2.BF16' in x for x in part)})
 static.append({'job':path.parent.name,'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'instructions':len(lines),'HGMMA':len(hs),'HMUL2_BF16':sum(' HMUL2.BF16' in x for x in lines),'between_two_MMA_group_windows':windows})
by={x['job']:x for x in static}
for m in [8,32]:
 a=by['w13_m%d_s4_control'%m];b=by['w13_m%d_grouped_tma_ws'%m]
 assert (a['HGMMA'],a['HMUL2_BF16'])==(b['HGMMA'],b['HMUL2_BF16'])
policy=D/'round2_25x/expert_partition/NATIVE_POLICY_COVERAGE_SWEEP.json';rows=[]
for r in json.loads(policy.read_text())['rows']:
 if r['arch']!='SM90' or 'stages' not in r:continue
 m=r['m'];counts=[(8*m)//256+int(e<(8*m)%256) for e in range(32)]
 for pattern,c in [('globally_balanced_rank0',counts),('all_local_eight_unique',[m]*8+[0]*24)]:
  for stage,n in [('w13',8192),('w2',4096)]:
   s=r['stages'][stage];bm=s['heavy_block'][0];light=s['light_block'][0];cutoff=r['cutoff'];k=4096
   for mode in ['scored_default','repaired_adaptive']:
    tiles=sum((v+(light if mode=='repaired_adaptive' and v<=cutoff else bm)-1)//(light if mode=='repaired_adaptive' and v<=cutoff else bm) for v in c)
    active=sum(v>0 for v in c);logical_nvfp4=tiles*n*k*9//16
    # Same M tile geometry and one conversion of each active expert; no cache/multicast assumptions.
    staged=(active*(n*k*9//16+2*n*k)+tiles*2*n*k)
    rows.append({'m':m,'pattern':pattern,'stage':stage,'mode':mode,'heavy_bm':bm,'light_bm':light,'cutoff':cutoff,'active_experts':active,'M_tile_weight_traversals':tiles,'decoded_weight_values':tiles*n*k,'logical_weight4_plus_scale_bytes':logical_nvfp4,'single_stage_packedBF16_plus_write_plus_GEMMread_bytes':staged,'staged_to_weight4_logical_payload_ratio':staged/logical_nvfp4 if logical_nvfp4 else None,'conversion_reuse_factor':tiles/active if active else None})
# New narrow alternative: exact exponent-corrected FP4 bits in registers.
# Native generic conversion yields q/4 before group scale; signed zero retained.
q=[0.,.5,1.,1.5,2.,3.,4.,6.];lut=[struct.unpack('<I',struct.pack('<f',v/4))[0]>>16 for v in q]
assert lut==[0,0x3e00,0x3e80,0x3ec0,0x3f00,0x3f40,0x3f80,0x3fc0]
def prmt(a,b,sel):
 vals=[(a>>(8*i))&255 for i in range(4)]+[(b>>(8*i))&255 for i in range(4)]
 return sum((vals[(sel>>(4*i))&7] if not((sel>>(4*i))&8) else (255 if vals[(sel>>(4*i))&7]&128 else 0))<<(8*i) for i in range(4))
def pair_lookup(val):
 # Two selected FP4 nibbles reside at bits12..15 and28..31, as in existing dequant shift.
 select=((val>>12)&7)|((val>>20)&0x700)
 lo=prmt(0xc0800000,0xc0804000,select)&0x00ff00ff
 hi=prmt(0x3e3e3e00,0x3f3f3f3f,select)&0x00ff00ff
 return lo|(hi<<8)|(val&0x80008000)
checks=0
rng=random.Random(81)
for a in range(16):
 for b in range(16):
  for _ in range(16):
   val=(rng.getrandbits(32)&~0xf000f000)|(a<<12)|(b<<28)
   out=pair_lookup(val);ref=(lut[a&7]|((a&8)<<12))|((lut[b&7]|((b&8)<<12))<<16)
   assert out==ref;checks+=1
report={'scope':'CPU source and existing SASS instruction accounting only; no runtime bottleneck attribution or speed prediction.', 'sources':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [policy]+[H/y for y in ['mma/wgmma.cuh','datatype/dequant.cuh','datatype/dequant_single.cuh','arith/mainloop_arith.cuh','arith/exp_offset.cuh','kernel/humming.cuh','kernel/humming_ws.cuh']]}, 'static_sass':static,'source_findings':['Indexed nonWS and compact TMA/WS both call the identical MMA transform_b and MainloopArithmetic methods. TMA transfers packed B4/scales into shared memory; consumer register-side unpack, FP4 exponent compensation and group scaling remain.','For admitted BN256/WN32/WK64/BK64 A16 geometry, transform_b has2 transforms of8 BF16 values per thread per16-K iteration. Four iterations and256 math threads account for16384=BN*BK weight values per M tile.','Observed native paired-MMA windows contain18 BF16 pair multiplies per thread:8 weight exponent corrections,8 group multiplies and2 surviving scale exponent corrections; windows occasionally17/19 due compiler movement. This is not18 cycles and does not count tensor work as scalar instructions.','Moving M/N tile order preserves every dequant transform and logical packed-weight request. Cache traffic can change; dequant ALU count does not. Local M3012 G2 negative is compatible with either weak cache benefit or other bottlenecks, and does not identify which.','Compact TMA/WS can reduce per-thread address/load issue and overlap producers with consumers. It does not itself reuse decoded B across separate M tiles. Existing SASS whole-static counts decrease, but producer/consumer execution masks and loop trips differ; no speed ratio follows.'], 'logical_payload_model':{'nvfp4_per_weight_bytes':.5625,'BF16_per_weight_bytes':2,'same_tile_count_large_reuse_payload_ratio_limit':32/9,'formula':'Original=T*NK*(1/2+1/16). One-time active-expert staging=Eactive*NK*(1/2+1/16+2)+T*NK*2. Excludes activation/C/glue and cache effects. All quantities are logical requested payload, not DRAM bytes.','rows':rows},'new_register_lookup_hypothesis':{'status':'CPU bit-pattern prototype only; uncompiled','checks':checks,'reference_values':'Native exponent-corrected q/4, all16 FP4 encodings including signedzero','mapping':lut,'operation_outline':'Extract2 selectors; two PRMT byte tables; mask/interleave and preserve signs. Replace generic FP4 conversion plus weight exponent HMUL2 only. Keep existing scale conversion, BF16 group multiply, MMA/K partition and epilogue4.','potential':'Avoid8 of18 HMUL2 pair instructions per16-weight transform without expanded global weights. Additional integer/PRMT cost may erase savings.','guard':'BF16A/C,FP4E2M1B,FP8E4M3GS16,nozeropoint,noinputscale,noF16acc,no native mixed,offsetx124. Cannot apply to other scale/exponent configurations.','next_decisive_gate':'Compile original/lookup same metadata and inspect actual instruction delta/registers, then full operand/native numerical gate. No native lease requested in this CPU report.'},'qualification':'Native H200 timing and exact correctness remain required; no serving gain established.'}
(P/'NATIVE_WEIGHT_WORK_AND_DEQUANT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'static_kernels':len(static),'traffic_rows':len(rows),'lookup_checks':checks,'m3012':[x for x in rows if x['m']==3012 and x['stage']=='w13' and x['mode']=='scored_default']}))

