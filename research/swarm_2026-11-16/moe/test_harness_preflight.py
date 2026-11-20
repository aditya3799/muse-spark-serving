from pathlib import Path
import ast,hashlib,json,os,sys
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p))
from probe_preflight import preflight,FROZEN
os.environ['MAX_JOBS']='1';os.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'
result=preflight(p.parents[1]/'sglang')
os.environ['MAX_JOBS']='2'
try:preflight(p.parents[1]/'sglang');raise AssertionError('unsafe compiler count accepted')
except RuntimeError:pass
os.environ['MAX_JOBS']='1';original=FROZEN['muse_spark_fp4_moe.py'];FROZEN['muse_spark_fp4_moe.py']='0'*64
try:preflight(p.parents[1]/'sglang');raise AssertionError('wrong source hash accepted')
except RuntimeError:pass
finally:FROZEN['muse_spark_fp4_moe.py']=original
files=['probe_preflight.py','frozen_helper_probe.py','memory_probe.py','launch_frozen_probe.py','launch_memory.py']
hashes={}
for name in files:
 f=p/name;ast.parse(f.read_text(encoding='utf-8-sig'));hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest()
result.update(status='native audit passed; GPU execution pending',harness_hashes=hashes,native_audit_python=sys.executable,linux_python='/home/aditya/.venvs/okc-marlin-tiles/bin/python',linux_nvcc_last_observed='/home/aditya/.venvs/okc-marlin-tiles/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc',negative_checks=['MAX_JOBS2 rejected','incorrect frozen SHA rejected'],fixes=['pin both frozen exports and independent baseline before GPU imports','release weight-generator lists before preparation','test eager outputs plus three replays including active/allremote transition','record every replay error and require exact zero on allremote','record prepare and graph peak reserved memory','report incomplete cooperative exits honestly','log exact runtime Python/nvcc executable paths'])
(p/'HARNESS_AUDIT.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({'status':result['status'],'hashes':hashes}))
