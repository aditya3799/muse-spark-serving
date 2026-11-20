import json,hashlib
from pathlib import Path
from evaluate_h200_receipt import evaluate,PARENT,SGLANG
here=Path(__file__).resolve().parent
r=dict(schema_version=1,actual_h200_execution=False,baseline_sha256=PARENT,sglang_commit=SGLANG,
  purpose='Current readiness from completed local/source evidence. Not an execution receipt.',
  completed_offline_evidence={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ['finite_frozen/manifest.json','finite_lifetime_v2/manifest.json','split_readiness_frozen/manifest.json','facade_memory.json','tiled_holdout_evaluation.json']},
  missing='Actual H200 coefficient/fullFFN/capture/memory/token and full-serving independent2x evidence')
(here/'h200_current_readiness.json').write_text(json.dumps(r,indent=2))
out=evaluate(r);(here/'h200_current_decision.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(decision=out['decision'],missing_gates=len(out['failing_gates']),scored_submission_authorized=out['scored_submission_authorized'])))
