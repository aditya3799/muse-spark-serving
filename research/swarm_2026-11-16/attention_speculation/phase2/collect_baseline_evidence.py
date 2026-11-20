"""CPU read-only provenance/log extraction; no API or runtime imports."""
import hashlib,json,re
from pathlib import Path
P=Path(__file__).resolve().parent;OWN=P.parent;SWARM=OWN.parent;ROOT=SWARM.parent
ROUND=SWARM/'coordination/round5_humming_adaptive_fused'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result_path=SWARM/'coordination/status_e399c79bb1994630.json'
job=json.loads(result_path.read_text(encoding='utf-8-sig'));result=job['result']
assert job['status']=='done' and result['passed'] is True
patch=ROUND/'candidate.diff'
assert sha(patch)=='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
fa=ROUND/'candidate_sources/python/sglang/srt/layers/attention/flashattention_backend.py'
metadata=ROOT/'sglang/python/sglang/kernels/ops/attention/metadata.py'
assert sha(fa)=='2a720c8f5ed86437e6ec181cace03ae97fba0460672b1d001549f8df4b71e03c'
assert sha(metadata)=='8aa5db57763f5ecfb01a05beba74791d0bf3921952c789d8a8432abf25c231ca'
log=result['startup_log']['text'];fields={}
for name in ('attention_backend','speculative_draft_attention_backend','kv_cache_dtype','speculative_algorithm','speculative_num_draft_tokens','speculative_num_steps','speculative_eagle_topk','enable_unified_memory','enable_page_major_kv_layout','enable_hierarchical_cache','disable_cuda_graph_padding','enable_two_batch_overlap','enable_dp_attention','tp_size','ep_size','page_size','chunked_prefill_size'):
    fields[name]=list(dict.fromkeys(re.findall(r"\b"+re.escape(name)+r"(?:=|':\s*)([^,\n)]+)",log)))
lines=[s for s in log.splitlines() if any(x in s for x in ('Draft metadata glue','Engine startup timings','KV cache allocated','Draft attention backend'))]
files=[fa,metadata]+[ROOT/'sglang/python/sglang/srt'/p for p in ('model_executor/runner/metadata_glue_graph.py','model_executor/runner/decode_cuda_graph_runner.py','mem_cache/kv_cache_configurator.py','mem_cache/kv_index_translator.py','speculative/dflash_worker_v2.py')]
optional=OWN/'combined_attention_optional_v1/manifest.json'
evidence={'status':'READ_ONLY_BASELINE_ANALYSIS','job_id':job['id'],'patch_sha256':sha(patch),'scored_result_sha256':sha(result_path),'frontier':result['frontier'],'rungs':result['rungs'],'gpu_memory':result['gpu_memory'],'startup_config_fields':fields,'selected_startup_lines':lines,'source_hashes':{str(p):sha(p) for p in files},'optional_manifest':{'path':str(optional),'sha256':sha(optional)},'remaining_uniform_gain_to_1_6':1.6/result['frontier']['score'],'remaining_uniform_time_reduction':1-result['frontier']['score']/1.6,'limits':['Aggregate logs provide no operator latency attribution.','Startup graph capture duration is not inference step latency.','Smaller concurrency frontier gains do not prove a CPU bottleneck.','No submission, GPU/compiler call, source modification or precision/policy change.']}
with (P/'BASELINE_EVIDENCE.json').open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(evidence,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'score':result['frontier']['score'],'startup_config_fields':fields,'selected_startup_lines':lines},indent=2,ensure_ascii=True))
