import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=next(p for p in H.parents if (p/'sglang/python/sglang/srt').is_dir());metric=json.loads((H/'METRIC_AND_WORKLOAD_AUDIT.json').read_text());manifest=json.loads((H/'TARGET_ARGMAX_OVERLAY_MANIFEST.json').read_text())
paths=['sglang/python/sglang/srt/speculative/dflash_worker_v2.py','sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py','sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py','sglang/python/sglang/srt/layers/logits_processor.py','sglang/python/sglang/srt/distributed/parallel_state.py','sglang/python/sglang/srt/utils/device_timer.py']
records=[]
for rel in paths:
 p=R/rel;raw=hashlib.sha256(p.read_bytes()).hexdigest();prior=metric['source_sha256'].get(rel);lf=hashlib.sha256(p.read_text(encoding='utf-8-sig').encode()).hexdigest();overlayprior=next((f['source_lf_normalized_sha256'] for f in manifest['files'] if 'sglang/python/'+f['path']==rel),None)
 if prior:assert raw==prior,(rel,raw,prior)
 if overlayprior:assert lf==overlayprior,(rel,lf,overlayprior)
 records.append({'path':rel,'actual_file_sha256':raw,'lf_normalized_sha256':lf,'prior_early_round4_file_hash_match':None if prior is None else True,'original_renderer_input_lf_hash_match':None if overlayprior is None else True})
report={'status':'BASELINE_INPUTS_UNCHANGED_WHERE_EARLY_HASH_AVAILABLE','files':records,'scope':'Read-only current file hashes; active source was not written by this lane. Early raw hashes or initial renderer content hashes are compared explicitly, absent baseline markednull.','research_only_outputs':['target_argmax_overlay','target_argmax_overlay_v2','device_timer_overlay']}
(H/'OVERLAY_BASELINE_INPUT_HASHES.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
