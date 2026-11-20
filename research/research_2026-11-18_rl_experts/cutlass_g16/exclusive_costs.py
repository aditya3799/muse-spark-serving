"""Exact bytes and a source-level 32-lane direct-register-gather transaction count, without timing."""
import json
from pathlib import Path
E=32
mib=2**20
cases=[]
for m in (1,8,32,128,256,768,3072,16384):
    active=E*(1-(1-8/256)**m)
    cases.append(dict(verify_or_prefill_rows=m,expected_uniform_active=active,read_write_MiB=54*active,
                      all_layers_floor_ms_if_crossing_HBM=60*54*active*mib/4.8e12*1000))
gathers=[]
for n in (4096,8192):
    for wgmma in (False,True):
        sector,totals=set(),[]
        for j in range(4):
            used=set()
            for v in range(32):
                lane=v//4
                qn=(v%4)*16+lane//4+(j%2 if wgmma else j//2)*8
                kp=lane%4+(j//2 if wgmma else j%2)*4
                used.add((qn*2048+kp)//32)
            sector.update(used);totals.append(len(used))
        gathers.append(dict(N=n,K=4096,WGMMA=wgmma,four_byte_instruction_sectors=totals,
                            distinct_sectors=len(sector),contiguous_native_u32_sectors=4))
out=dict(large_GEMM_weight_repack_per_forward=0,steady_resident_MiB_per_rank=60*864,
         additional_persistent_weights_MiB=0,exclusive_conversion_peak_per_layer_MiB=864,
         per_stream_native_workspace_MiB=576,grouped_activation_extra_MiB=12,
         expected_uniform_assumption="Eight distinct global routes chosen uniformly and independently per input row",
         full_restore_MiB=1728,cases=cases,direct_gather=gathers)
Path(__file__).with_name("exclusive_costs.json").write_text(json.dumps(out,indent=2))
print([(r['verify_or_prefill_rows'],round(r['expected_uniform_active'],1),round(r['read_write_MiB'],1)) for r in cases])
print(gathers[0])
