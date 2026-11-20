"""CPU audit of ragged row eligibility against frozen-release graph dispatch."""
import ast
import bisect
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'sglang/python/sglang/srt/arg_groups/cuda_graph_hook.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_prefill_cuda_graph_batch_sizes')
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), ns)
buckets = ns[fn.name](16384)
assert all(m % 8 == 0 for m in buckets if m >= 512)
results=[]
for m in (1,4,16,513,751,753,755,3011,3012,3013,16383,16384):
    padded=buckets[bisect.bisect_left(buckets,m)]
    results.append(dict(raw=m,padded=padded,raw_ragged=m>=512 and m%8!=0,
                        graph_row_projection_eligible=padded>=512 and padded%8==0))
startup=json.loads((ROOT/'release_2026-11-18/scored_submission_status.json').read_text(encoding='utf-8'))['result']['startup_log']['text']
capture_evidence=[s for s in startup.splitlines() if 'Capture target prefill CUDA graph end' in s]
assert capture_evidence
report=dict(baseline_sha256=hashlib.sha256((ROOT/'release_2026-11-18/candidate.diff').read_bytes()).hexdigest(),
            bucket_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            buckets=buckets,cases=results,startup_evidence=capture_evidence,
            decision='Eager fallback only; every default graph bucket >=512 is divisible by TP8. No scored workload eager-frequency evidence, so no serving speedup claim.',
            limits=['Assumes default generated buckets; stored startup server_args line is truncated.',
                    'Graph runner can fall back for metadata/context/backend restrictions; actual fallback rate is unknown.'])
(HERE/'graph_applicability.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
