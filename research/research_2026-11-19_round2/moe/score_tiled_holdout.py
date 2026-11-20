import json,statistics
from pathlib import Path
here=Path(__file__).resolve().parent;d=json.loads((here/'tiled_holdout_930349.json').read_text());plan=json.loads((here/'tiled_holdout_plan.json').read_text());rows=[]
for c in d['cases']:
 b=c['samples_us']['baseline'];a=c['samples_us']['candidate'];r=c['samples_us']['control'];ratios=[x/y for x,y in zip(b,r)]
 valid=min(ratios)>=.95 and max(ratios)<=1.05
 speed=min(statistics.median(b),statistics.median(r))/statistics.median(a)
 rows.append(dict(m=c['m'],mode=c['mode'],captured=c['capture'],correct=c['correct'],control_min=min(ratios),control_max=max(ratios),control_valid=valid,conservative_component_ratio=speed,qualified_local_component=valid and speed>1.10))
report=dict(complete=True,plan=plan,cases=rows,qualified=sum(x['qualified_local_component'] for x in rows),scope='Generic restoration+completeFFN versus improved restoration+sameFFN only. Not a serving speedup, not faster than canonical frozen parent. No H200 execution.',no_further_repeat=True)
(here/'tiled_holdout_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
