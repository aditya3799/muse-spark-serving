from pathlib import Path
import hashlib,json,ast
P=Path(__file__).resolve().parent;H=P.parent/'hopper_moe'
def one(s,a,b):
 assert s.count(a)==1,(a,s.count(a))
 return s.replace(a,b)
cu=(H/'swap_ab_probe.cuh').read_text(encoding='utf-8-sig')
cu=one(cu,'template<bool Swap> __device__ void run','template<bool Swap,int Wrong=0> __device__ void run')
cu=one(cu,'// Bank b visits BK*outer + 64*b + 16*iter. Keep this order before final bank0+bank1.','// Four banks retain256*outer+64*bank+16*iter; pinned tree is(0+2)+(1+3).')
cu=one(cu,'float v=c[0][t][r];if constexpr(BANKS==2)v=__fadd_rn(v,c[1][t][r]);','''static_assert(BANKS==4);
   float v;
   if constexpr(Wrong==1)v=__fadd_rn(__fadd_rn(c[0][t][r],c[1][t][r]),__fadd_rn(c[2][t][r],c[3][t][r]));
   else if constexpr(Wrong==2)v=__fadd_rn(__fadd_rn(__fadd_rn(c[0][t][r],c[1][t][r]),c[2][t][r]),c[3][t][r]);
   else v=__fadd_rn(__fadd_rn(c[0][t][r],c[2][t][r]),__fadd_rn(c[1][t][r],c[3][t][r]));''')
cu+='\nextern "C" __global__ void wrong_adjacent(const uint16_t*x,const uint32_t*w,const uint8_t*s,const int32_t*slots,uint16_t*out,float*raw,int capacity){run<true,1>(x,w,s,slots,out,raw,capacity);}\nextern "C" __global__ void wrong_left_fold(const uint16_t*x,const uint32_t*w,const uint8_t*s,const int32_t*slots,uint16_t*out,float*raw,int capacity){run<true,2>(x,w,s,slots,out,raw,capacity);}\n'
(P/'four_bank_probe.cuh').write_text(cu)
py=(H/'swap_ab_probe.py').read_text(encoding='utf-8-sig').replace("k not in (128,4096) or banks not in (1,2)","k not in (256,4096) or banks!=4").replace("'swap_ab_probe.cuh'","'four_bank_probe.cuh'")
py=one(py,"('swap_ab','standard_mma','unpack_operand')","('swap_ab','standard_mma','unpack_operand','wrong_adjacent','wrong_left_fold')")
py=one(py,"assert name in ('swap_ab','standard_mma')","assert name in ('swap_ab','standard_mma','wrong_adjacent','wrong_left_fold')")
py=one(py,"16 if name=='swap_ab' else 8","8 if name=='standard_mma' else 16")
(P/'four_bank_probe.py').write_text(py)
config=(H/'SM86_COMPACT_E8_W13_JOBS.json').read_bytes();(P/'FOUR_BANK_REFERENCE_CONFIG.json').write_bytes(config)
s=(H/'probe_swap_ab.py').read_text(encoding='utf-8-sig')
s=s.replace('from swap_ab_probe import Probe','from four_bank_probe import Probe')
s=s.replace("P/'swap_ab_probe.py',P/'swap_ab_probe.cuh',P/'test_swap_ab_cpu.py',P/'SWAP_AB_CPU_CONTRACT.json',P/'SM86_COMPACT_E8_W13_JOBS.json'","P/'four_bank_probe.py',P/'four_bank_probe.cuh',P/'test_four_bank_reducer.py',P/'FOUR_BANK_REDUCER_CONTRACT.json',P/'FOUR_BANK_REFERENCE_CONFIG.json'")
s=s.replace("P/'SM86_COMPACT_E8_W13_JOBS.json'","P/'FOUR_BANK_REFERENCE_CONFIG.json'")
s=one(s,'for k in [128,4096]:','for k in [256,4096]:')
s=one(s,'for banks in [1,2]:','for banks in [4]:')
s=one(s,'block_shape=[16,256,64*banks]','block_shape=[16,128,64*banks]')
s=one(s,"['swap_ab','standard_mma','pinned']","['swap_ab','standard_mma','pinned','wrong_adjacent','wrong_left_fold']")
s=one(s,"for name in ['swap_ab','standard_mma']","for name in ['swap_ab','standard_mma','wrong_adjacent','wrong_left_fold']")
s=one(s,'==5','==9')
s=one(s,'[0,1,7,8,9,15,16,17,23,24,9,0]','[0,1,7,8,9,15,16,17,23,24,9,0,8,24]')
anchor='    for t in list(out.values())+list(raw.values()):t.fill_(37.)'
insert='''    engineered=generation>=12
    if engineered:
     # q4(code6),scale1 makes quarterweight1. One nonzero product in each64-widebank.
     canonical.fill_(0x66666666);layer.weight.fill_(0x66666666);layer.weight_scale.fill_(1.)
     x.zero_()
     partials=[2**24,1,-2**24,1] if generation==12 else [2**24,-2**24,1,1]
     for bank,value in enumerate(partials):x[:,64*bank]=value
    for t in list(out.values())+list(raw.values()):t.fill_(37.)'''
s=one(s,anchor,insert)
anchor="    row=dict(k=k,banks=banks,generation=generation,count=count,swap_vs_standard=a,standard_vs_pinned=b,fp32_swap_vs_standard=f,external_bf16_postscale_swap=post_swap,external_bf16_postscale_pinned=post_native,sentinel=sentinel);r['rows'].append(row);save()"
insert='''    engineered_checks=None
    if engineered:
     expected=2. if generation==12 else 1.;wrong=1. if generation==12 else 2.
     engineered_checks=dict(expected_raw=expected,expected_bf16_x4=expected*4,raw_known=all(bool((raw[name][active]==expected).all())for name in ('swap_ab','standard_mma')),post_known=all(bool((out[name][active]==expected*4).all())for name in ('swap_ab','standard_mma','pinned')),wrong_adjacent_reproduced=bool((raw['wrong_adjacent'][active]==wrong).all()),wrong_left_fold_reproduced=bool((raw['wrong_left_fold'][active]==wrong).all()))
    row=dict(k=k,banks=banks,generation=generation,count=count,swap_vs_standard=a,standard_vs_pinned=b,fp32_swap_vs_standard=f,external_bf16_postscale_swap=post_swap,external_bf16_postscale_pinned=post_native,sentinel=sentinel,engineered_checks=engineered_checks);r['rows'].append(row);save()
    if engineered:assert all(engineered_checks[z]for z in ('raw_known','post_known','wrong_adjacent_reproduced','wrong_left_fold_reproduced')),row
    assert a['bitwise']and f['bitwise']and post_swap['bitwise'],row'''
s=one(s,anchor,insert)
s=s.replace("scope='E1 N256 direct-global canonical WGMMA-packed NVFP4 swap-AB versus ordinary MMA and pinned standard Humming; same precision, no pipeline/performance claim, no Hopper equivalence claim.'","scope='Four-bank E1 N256 K256/4096 direct-global canonical packed arithmetic microgate; pinned ordinary MMA comparator BN128/BK256/WK64 (two output tiles), not native SM90 WGMMA. Engineered bank sums raw2/x4=8 and raw1/x4=4; negative adjacent/left-fold kernels. No pipeline, performance, or H200 claim.'")
s=s.replace("status='COMPLETE_MAPPING_GATE'","status='COMPLETE_FOUR_BANK_ARITHMETIC_GATE'")
(P/'probe_four_bank.py').write_text(s)
launcher=(H/'launch_probe.py').read_text(encoding='utf-8-sig').replace("default='profile_humming_stages.py'","default='probe_four_bank.py'")
(P/'launch_four_bank.py').write_text(launcher)
for f in ('four_bank_probe.py','probe_four_bank.py','launch_four_bank.py'):ast.parse((P/f).read_text())
r={'status':'SOURCE_PREPARED_NOT_COMPILED','upstream_sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [H/'swap_ab_probe.cuh',H/'swap_ab_probe.py',H/'probe_swap_ab.py',H/'launch_probe.py']},'derived_sources':{str(P/f):hashlib.sha256((P/f).read_bytes()).hexdigest()for f in ('four_bank_probe.cuh','four_bank_probe.py','probe_four_bank.py','launch_four_bank.py')},'scope':'Copied isolated direct-global arithmetic probe only. Current Hopper two-bank implementation unchanged. Future lease required.'}
(P/'FOUR_BANK_MICROPROBE_DERIVATION.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'])
