import pathlib,json,hashlib,numpy as np
from packed_expert_slots import Bank,pack_codes,unpack_codes,transform_scales,V,PORT
P=pathlib.Path(__file__).parent;cases=[];comparisons=0;neg=[]
for trans in (False,True):
 for h,i in [(64,64),(128,64),(64,128)]:
    base=Bank(4,h,i,trans);logical={}
    for stage,n,k in [('w13',2*i,h),('w2',h,i)]:
        e,rr,kk=np.indices((4,n,k));codes=((e*3+rr*7+kk*5+kk//16+rr//8)%16).astype(np.uint8)
        e,rr,gg=np.indices((4,n,k//16));scales=((e*43+rr*13+gg*17)%256).astype(np.uint8)
        post=np.array([0x3f80,0xbf80,0x3f00,0x4000],np.uint16)
        base.install(stage,codes,scales,post);logical[stage]=(codes,scales,post)
        # Actual source scale function preserves expert independence.
        assert np.array_equal(transform_scales(scales),np.concatenate([transform_scales(scales[j:j+1])for j in range(4)]))
        assert np.array_equal(unpack_codes(base.tensors[stage+'_weight'],n,k,trans),codes)
        comparisons+=codes.size+scales.size
    for cap in [33,40,64]:
        dst=Bank(cap,h,i,trans)
        for slot,expert in [(0,3),(1,0),(cap-1,2)]:
            before={k:v.copy()for k,v in dst.tensors.items()};dst.copy_slot(slot,base,expert)
            for key,v in dst.tensors.items():
                assert np.array_equal(v[slot],base.tensors[key][expert]);mask=np.arange(cap)!=slot
                assert np.array_equal(v[mask],before[key][mask])
            for stage,n,k in [('w13',2*i,h),('w2',h,i)]:
                decoded=unpack_codes(dst.tensors[stage+'_weight'][slot:slot+1],n,k,trans)
                assert np.array_equal(decoded[0],logical[stage][0][expert])
                assert np.array_equal(dst.tensors[stage+'_weight_scale'][slot],transform_scales(logical[stage][1][expert:expert+1])[0])
                assert dst.tensors[stage+'_post_scale'][slot]==logical[stage][2][expert]
                comparisons+=n*k+n*k//16+1
        cases.append(dict(transpose=trans,H=h,I=i,capacity=cap,slot_bytes=dst.bytes_per_slot()))
    try:Bank(3,h,i,not trans).copy_slot(2,base,0)
    except ValueError:neg.append('cross_arch_layout')
    else:raise AssertionError('negative')
# Native-shape sizes from the exact same slot formulas, no giant allocation.
h=i=4096;slot_bytes=(3*h*i//2)+(3*h*i//16)+4
report={'status':'PASS_PINNED_PACK_LAYOUT_EXPERT_ROW_COPY_CPU','cases':cases,'element_checks':comparisons,'negative_cases':neg,'native_4096_slot_bytes':slot_bytes,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [V/'include/humming/kernel/process.cuh',V/'ops/weight.py',V/'transform.py',PORT,P/'packed_expert_slots.py',P/'test_packed_expert_slots.py']},'scope':'Literal pinned A16/B4 interleave3 pack loops plus independent direct inverse; actual source-extracted FP8-scale transform with Numpy tensor shim. Both SM86 and SM90 mini-block orientations. Copies complete packed/scales/BF16-postscale rows without re-quantization. No GPU, peer-copy, actual Humming launch or FP arithmetic proof.'}
(P/'PACKED_SLOT_COPY_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'cases':len(cases),'element_checks':comparisons,'native_slot_bytes':slot_bytes}))
