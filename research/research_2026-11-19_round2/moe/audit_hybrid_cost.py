import json,math
from pathlib import Path
here=Path(__file__).resolve().parent
packed=32*(8192+4096)*4096*9//16
traffic=packed*2
rows=[]
for m in (8,128,256,4096,4097,16384):
 chunks=math.ceil(m/2048)
 rows.append(dict(rows=m,normal_chunks=chunks,packing_passes_per_layer=1,unique_bytes_read_plus_written=traffic,amortized_unique_conversion_bytes_per_chunk=traffic/chunks,naive_repack_each_chunk_bytes=traffic*chunks))
normal=json.loads((here.parents[1]/'research_2026-11-19/moe/normal_package/coefficient_memory_audit.json').read_text())
report=dict(scope='Unique source/destination conversion traffic lower bound, not physical transactions or runtime prediction',one_rank_packed_layer_bytes=packed,one_conversion_bytes=traffic,rows=rows,per_model_pass_conditional_60_quantized_layers_bytes=60*traffic,per_model_pass_if_all_62_layers_quantized_bytes=62*traffic,endpoint_dtype_unverified=True,per_turn_formula='conversion_bytes * converted_layer_calls_per_turn; every decode invocation would pay again if fallback selected',scratch=dict(normal_torch_arena=normal['torch_arena_bytes_excluding_CABI'],shared_packed=packed,normal_nonpacked=normal['torch_arena_bytes_excluding_CABI']-packed,facade_locks=8192,combined_torch_owned_if_shared=normal['torch_arena_bytes_excluding_CABI']+8192,naive_duplicate_packed_total=normal['torch_arena_bytes_excluding_CABI']+packed+8192,native_wgs_tiles=(2*(384*8+8*(32+384*8//64+8))*4096*2),notes='Combined ownership is not serving peak. Add native/CUTLASS handles, WGS counters, live inputs/outputs and graph-private pools. Native FFN temporaries and normal scratch must be mutually reused/released where lifetimes allow. Existing E32 fixture only proved fallback delta864MiB.'),timing_file='fallback_930293.json',timing_hardware='RTX3050 SM86; never extrapolate toH200',decision='Fallback compatibility exact; reject restoration as decode optimization. Normal conversion can be amortized over large prefill chunks, but H200 grouped results required.')
(here/'hybrid_traffic_memory.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
