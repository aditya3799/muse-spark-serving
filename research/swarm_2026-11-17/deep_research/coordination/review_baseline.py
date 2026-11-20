import ast, hashlib, json, math, pathlib, re
ROOT = pathlib.Path(r'D:\OpenKernelChallenge\serving')
OUT = ROOT / 'swarm_2026-11-17/deep_research/coordination'
job = json.loads((OUT / 'remote_e399c79bb1994630.json').read_text(encoding='utf-8-sig'))
image = {1:3.708, 4:3.784, 16:3.934, 32:3.829}
points = job['result']['frontier']['points']
rows = []
for point in points:
    target = image[point['concurrency']]
    factor = target / point['ratio']
    rows.append({'concurrency':point['concurrency'], 'our_score_ratio':point['ratio'], 'image_ratio':target, 'additional_ratio_required':factor, 'fixed_work_remaining_time_fraction':1/factor, 'fixed_work_time_reduction_required':1-1/factor})
text = job['result']['startup_log']['text']
source = ROOT / 'sglang/python/sglang/srt/arg_groups/cuda_graph_hook.py'
tree = ast.parse(source.read_text(encoding='utf-8'))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_prefill_cuda_graph_batch_sizes')
module = ast.Module(body=[node], type_ignores=[])
ns = {}
exec(compile(module, str(source), 'exec'), ns)
buckets = ns[node.name](16384)
padding = [{'raw_rows':n,'bucket':min(b for b in buckets if b>=n),'padding_fraction':min(b for b in buckets if b>=n)/n-1} for n in [511,512,513,753,768,1000,1024,1506,2049,3012,4097,8193,16000,16384]]
result = {
 'status':'MEASURED_BASELINE_AND_SOURCE_AUDIT_NOT_SPEEDUP_FORECAST',
 'screenshot':{'reported_geomean':3.813,'geomean_from_rounded_rungs':math.prod(image.values())**.25,'scope':'User-provided screenshot; configuration and provenance unavailable'},
 'baseline_job':job['id'], 'baseline_score':job['result']['frontier']['score'], 'target_gap':rows,
 'interpretation':'Additional ratio is a target comparison, not an achievable forecast. Fixed-work reciprocal model does not model throughput-frontier interpolation.',
 'prefill_buckets':{'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'count':len(buckets),'rows':padding,'finding':'Existing dense bucket spacing makes generic padding-only changes too small to explain target gap; no workload-specific sizes proposed.'},
 'startup_evidence':[line for line in text.splitlines() if len(line)<1500 and ('captured 4662' in line or 'Capture target prefill CUDA graph end' in line or 'transfer_layer_id_max=' in line)],
 'new_remote_status':{'fa0d2a0fcf584339':'done_passed_check_c16_1.069_at_stock_point','6570fbd9e1b64ce4':'queued','new_submissions':0},
 'baseline_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'patch.diff',ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate.diff']}
}
(OUT/'SYSTEMS_BASELINE_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'gap':rows,'bucket_count':len(buckets),'padding':padding},indent=2))
