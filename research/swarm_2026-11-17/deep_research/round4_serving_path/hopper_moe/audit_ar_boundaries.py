import collections,hashlib,json,math,pathlib,re,struct
P=pathlib.Path(__file__).resolve().parent
def f32(v):return struct.unpack('<f',struct.pack('<f',v))[0]
def bf16(v):
 u=struct.unpack('<I',struct.pack('<f',v))[0];u=((u+0x7fff+((u>>16)&1))>>16)&65535
 return struct.unpack('<f',struct.pack('<I',u<<16))[0]
def rank_sum(v):
 z=v[0]
 for x in v[1:]:z=f32(z+x)
 return bf16(z)
a=[bf16(1),bf16(2**-8)]+[0.]*6;w=bf16(1.5)
original=bf16(rank_sum(a)*w);moved=rank_sum([bf16(x*w)for x in a]);assert original!=moved
v=[float(2**24),1.,float(-2**24),1.,0.,0.,0.,0.]
sequential=rank_sum(v);tree=bf16(f32(f32(v[0]+v[2])+f32(v[1]+v[3])));assert sequential!=tree
# Nonlinearity example, not an emulator of the original CUDA reduction layout.
def norm(v):
 scale=1/math.sqrt(sum(x*x for x in v)/len(v)+1e-6)
 return [bf16(x*scale) for x in v]
normal=norm([1.,2.]);early=[rank_sum([norm([1.,0.])[i],norm([0.,2.])[i]]) for i in range(2)];assert normal!=early
sass=(P/'offline_mc_01/kernel.sass').read_text(encoding='utf-8-sig')
mc=sass[sass.index('.text.stock_mc_reduce'):]
inst=[l.strip() for l in mc.splitlines() if any(x in l for x in ['LDGMC','MEMBAR','REDG.E.ADD','LDG.E.STRONG.SYS','BAR.SYNC'])]
result=dict(status='SOURCE_AND_CPU_COUNTEREXAMPLES_WITH_OFFLINE_SASS',counterexamples=dict(projection_before_AR=dict(rank_inputs=a,weight=w,original_AR_BF16_then_projection=original,moved_projection_before_AR=moved),rank_tree_change=dict(values=v,rank_order_FP32_then_BF16=sequential,alternate_pairtree=tree),normalize_before_AR=dict(original=normal,moved=early),skip_AR_round=dict(inputs=a,baseline_BF16=rank_sum(a),unrounded_FP32=f32(sum(a)))) ,native_MC_instructions=inst,native_MC_primitive_found='LDGMC.E.F32ADD.BF16x8.RN' in mc,native_SYS_release_barrier_found='MEMBAR.ALL.SYS' in mc,sass_sha256=hashlib.sha256((P/'offline_mc_01/kernel.sass').read_bytes()).hexdigest(),limitations=['No hardware multicast numerical emulation or runtime result','CPU examples reject algebraic transforms, not prove full norm parity','Software rank order cannot substitute for hardware MC merely because both accumulate FP32'])
assert result['native_MC_primitive_found'] and result['native_SYS_release_barrier_found']
(P/'AR_NUMERIC_BOUNDARIES_AND_SASS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
