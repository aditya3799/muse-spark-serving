"""Scope-labelled old-profile sensitivity; never a current serving forecast."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
profile = HERE.parents[2] / 'muse_prefill_layer/profiles/a1d2f61ed3534773.json'
data = json.loads(profile.read_text())
rows = []
for record in data['profiles']:
    case, report = record['case'], record['report']
    total = float(re.search(r'^total\s+([\d.]+)', report, re.M).group(1))
    busy_match = re.search(r'^GPU busy[^\n]*?\s+([\d.]+)\s', report, re.M)
    busy = float(busy_match.group(1)) if busy_match else None
    attention = []
    for line in report.splitlines():
        if 'okc_attn::attn_' in line:
            time = re.search(r'\s+([\d.]+)\s+[^\d]+[\d.]+%\s+\d+$', line)
            if time:
                attention.append(float(time.group(1)))
    rows.append(dict(case=case, kernel_sum_us=total, gpu_busy_us=busy,
                     displayed_attention_sum_us=sum(attention)))
first = [r for r in rows if r['case']['load'] == 'first turn']
weighted_sum = sum(r['case']['weight'] * r['kernel_sum_us'] for r in first)
full_attention_sum = sum(r['case']['weight'] * r['displayed_attention_sum_us']
                         for r in first if r['case']['layer'] == 'full')
share = full_attention_sum / weighted_sum
out = dict(
    scope='OLD custom kernel-track profiles, different MoE/dense paths, mocked collectives. Kernel sums include overlap/waits; displayed rows may omit small kernels. These arithmetic sensitivities are NOT current serving or nativeFA3 forecasts.',
    rows=rows,
    first_turn_weighted_kernel_sum_us=weighted_sum,
    first_turn_full_attention_displayed_sum_us=full_attention_sum,
    first_turn_displayed_full_attention_fraction=share,
    hypothetical_zero_cost_full_attention_speed_ratio=1/(1-share),
    memory=dict(kv_arena_bound=128*1024*1024, moe_owned_arena=905977856,
                persistent_sum=128*1024*1024+905977856,
                excludes='Native FFN/attention intermediates, allocator fragmentation, NCCL, other candidate scratch'),
)
(HERE / 'prefill_stage_budget.json').write_text(json.dumps(out, indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='rows'}, indent=2))
