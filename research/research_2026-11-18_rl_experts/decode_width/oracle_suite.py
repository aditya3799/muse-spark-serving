"""Run the known B32 mismatching case and forced crossings against dense-router oracle."""
import json
import sys
from pathlib import Path
sys.argv.extend(["--method","dense_histogram"])
import repair_widths as r

here=Path(__file__).resolve().parent
t=r.original.torch
report=dict(seed=98173,method="dense_histogram",complete=False,cases={})
def save():
    (here/"oracle_suite.json").write_text(json.dumps(report,indent=2))
with t.inference_mode():
    t.manual_seed(98173)
    t.set_num_threads(2)
    layer=r.original.layer_from_release()
    for b,pattern,forced,rounds in [(32,"uniform",False,4),(32,"skewed",True,2),(16,"skewed",True,2)]:
        r.custom.stress_crossing=forced
        r.install()
        case=r.original.evaluate(layer,b,98173,rounds,pattern)
        key=f"B{b}_{pattern}"+("_crossing" if forced else "")
        report["cases"][key]=case
        save()
        print(key,sum(v["bit_mismatches"] for v in case["rows"]),flush=True)
report["complete"]=True;save()
