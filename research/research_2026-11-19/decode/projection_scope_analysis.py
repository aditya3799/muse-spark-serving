"""Analytical work/unique-weight accounting, never a runtime or HBM profile."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
SERVING=HERE.parents[1]
H=8192;D=4096;I=4096;E=256;TOP=8;TP=8;LAYERS=62
FC=H*(5*H)
ATTN=(2*H+2*16*64)*H//TP+H*H//TP
EXPERT=3*D*I
PROJ=2*H*D
ROUTER=H*E
rows=[]
for m in [8,32,128,256,753,16384]:
 distinct=32*(1-(1-TOP/E)**m)
 lower=LAYERS*(ATTN+EXPERT*TOP/TP+PROJ/TP+ROUTER)
 upper=LAYERS*(ATTN+EXPERT*TOP/TP+PROJ+ROUTER)
 fixed_min=LAYERS*(2*ATTN+2*PROJ/TP+4*ROUTER)
 fixed_full=LAYERS*(2*ATTN+2*PROJ+4*ROUTER)
 packed_expert=EXPERT*(.5+1/16)
 experts_independent=LAYERS*distinct*packed_expert
 experts_same_balanced=LAYERS*packed_expert # same top8, one on every rank
 experts_same_one_rank=LAYERS*TOP*packed_expert # same top8, all on one rank
 dense_tokens=m*2*(ATTN+PROJ/TP+ROUTER)*LAYERS
 rows.append(dict(M=m,kind='decode_verify' if m<=256 else 'prefill_example',
  fc_GFLOP=2*m*FC/1e9,target_min_partitioned_GFLOP=2*m*lower/1e9,
  target_replicated_projection_GFLOP=2*m*upper/1e9,
  fc_share_matmul_work_max=FC/(FC+lower),fc_share_matmul_work_min=FC/(FC+upper),
  target_expert_balanced_GFLOP=2*m*LAYERS*EXPERT*TOP/TP/1e9,
  target_expert_worst_rank_GFLOP=2*m*LAYERS*EXPERT*TOP/1e9,
  independent_expected_distinct_local_experts_per_layer=distinct,
  unique_weight_GB=dict(fc=2*FC/1e9,fc_Nshard=2*FC/TP/1e9,
   fixed_target_min_partitioned=fixed_min/1e9,fixed_target_full_projection=fixed_full/1e9,
   independent_experts=experts_independent/1e9,
   maximally_correlated_balanced_experts=experts_same_balanced/1e9,
   maximally_correlated_one_rank_experts=experts_same_one_rank/1e9),
  fc_fraction_unique_weight_independent_max=2*FC/(2*FC+fixed_min+experts_independent),
  fc_fraction_unique_weight_correlated_balanced_max=2*FC/(2*FC+fixed_min+experts_same_balanced),
  output_N_gather_bytes_per_rank_receive=2*m*H*(TP-1)//TP,
  output_N_extra_contiguous_copy_min_read_write_bytes=4*m*H,
  fc_input_read_plus_output_write_bytes=2*m*(5*H+H)))
 if m<=256:
  rows[-1]['attention_context_scenarios']=[dict(prefix_tokens=L,
   qk_pv_GFLOP=4*16*64*m*(16*(L+4.5)+46*min(2048,L+4.5))/1e9,
   cache_unique_GB_once_per_queryblock=256*(m/8)*(16*(L+8)+46*min(L+8,2055))/1e9)
   for L in [2048,8192,74160]]
 else:
  sliding_pairs=sum(min(i,2048) for i in range(1,m+1))
  rows[-1]['attention_first_prefill_qk_pv_GFLOP']=4*16*64*(16*m*(m+1)/2+46*sliding_pairs)/1e9
sources=[]
for p in [SERVING/'release_2026-11-18/candidate_sources/python/sglang/srt/models/muse_spark.py',
 SERVING/'sglang/python/sglang/srt/configs/muse_spark.py',
 SERVING.parent/'muse_prefill_layer/reference.py',HERE.parent/'prefill/recovered_public_fc_receipt.json']:
 sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
doc=dict(scope=__doc__,architecture_source='https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B',
 architecture_access_utc='2026-11-19T10:08:00Z',target_checkpoint_sha='75271573ed1e94a216d2e69a988c3354a4b69208',
 actual_checkpoint_config_available=False,architecture_basis='Primary engine recipe:62MoE,zero dense,46SWA/16full. Expert dimensions cross-checked against official local challenge reference. SGLang defaults alone are not checkpoint evidence.',
 endpoint_expert_dtype='Unverified for actual checkpoint. All62 experts counted as packed4-bit+one8-bit scale/16coeff to avoid overstating target unique-weight traffic. If2endpointlayers BF16, add2*distinct*EXPERT*(2-.5625)bytes per rank.',
 assumptions=['TP8/EP8, replicated target token rows, balanced expected expert assignments; every token chooses8distinct experts. Independent-row uniform routing is a model, not observed distribution.',
 'Target MoE pre/post projections bracketed between ideal1/8 partition and full replication. This does not claim baseline enables both partitions on every shape.',
 'FLOPs count2per multiply-add, include QKV+gate/O, router, pre/postMoE, activeexperts. Omit attention QK/PV,norms,activations,LMhead,all draftlayers/contextKV,transport and scheduler. Omissions make reported FCfraction optimistic under equal FLOP-cost assumptions.',
 'Unique-weight bytes are tensor footprints under a once-per-step read model, not HBMtraffic or a latency lowerbound. Caching, splitK, rereads,epilogue,formatconversion can change actual traffic.',
 'FP32router FLOPs are counted as arithmetic only, not assigned BF16tensorcore cost. No bandwidth,efficiency or H200speed assumptions.',
 'Prefill rows753and16384 are illustrative actual rawtail/fullchunk sizes; Npackage only registers configureddecode rows. No Ncandidate prefillcoverage claimed. Rowpartition prefill reuses fullFCweight perrank and cuts M arithmetic, so itsFCweightfootprint remains671MB.'],
 dimensions=dict(hidden=H,expert_hidden=D,expert_intermediate=I,experts=E,top_k=TOP,layers=LAYERS,tp=TP,
  fc_coefficients=FC,attention_coefficients_per_rank_per_layer=ATTN,expert_coefficients=EXPERT),
 baseline_score=1.2363,component_speedup=4.32002979296068,
 fraction_required_for_2x_if_uniform=(1-1.2363/2)/(1-1/4.32002979296068),
 conclusion='The4.32xcomponent is not evidenceof2xserving. Architecture work and footprint shares suggest projectionalone is unlikely to supply required~49.7%parent time, but they do not prove a runtimeupperbound. ActualH200full-frontiermeasurements required.',
 sources=sources,rows=rows)
(HERE/'projection_scope_analysis.json').write_text(json.dumps(doc,indent=2))
for r in rows:print(r['M'],round(r['fc_GFLOP'],3),round(r['target_min_partitioned_GFLOP'],3),round(r['independent_expected_distinct_local_experts_per_layer'],3),round(r['fc_fraction_unique_weight_independent_max']*100,3))
print('FC arithmetic fraction',rows[0]['fc_share_matmul_work_min'],rows[0]['fc_share_matmul_work_max'],'required',doc['fraction_required_for_2x_if_uniform'])
