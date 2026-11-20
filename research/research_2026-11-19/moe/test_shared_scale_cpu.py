"""Execute the actual Triton interpreted shared-scale lifecycle, no GPU allocation."""
from pathlib import Path
import json,os,sys,hashlib,time
here=Path(__file__).resolve().parent
old=here.parents[1]/'research_2026-11-18_rl_experts/cutlass_g16'
os.environ['TRITON_INTERPRET']='1'
sys.path.insert(0,str(old))
from test_shared_scales import check
start=time.monotonic()
results=[check(s) for s in (71273,89459)]
report={'scope':'Actual existing shared-scale Triton functions under CPU interpreter; no GPU timing claim','results':results,'elapsed_s':time.monotonic()-start,'source_sha256':{n:hashlib.sha256((old/n).read_bytes()).hexdigest() for n in ('shared_scales.py','exclusive_pack.py','group_prep.py','test_shared_scales.py')}}
(here/'shared_scales_cpu.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
