import hashlib,json,subprocess
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1];pin='297e54fd105e974cdf401e5b4d20253a22c16f90'
files=['python/sglang/srt/speculative/spec_tp_sync.py','python/sglang/srt/distributed/parallel_state.py','python/sglang/srt/distributed/device_communicators/pynccl.py','python/sglang/srt/managers/scheduler.py','python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py','python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py','python/sglang/srt/layers/dp_attention.py','python/sglang/kernels/ops/speculative/dflash.py']
out=here/'sync_sources';out.mkdir(exist_ok=True);records=[]
for name in files:
 blob=subprocess.check_output(['git','-C',str(root/'sglang'),'show',pin+':'+name]);path=out/Path(name).name;path.write_bytes(blob)
 records.append(dict(path=str(path.relative_to(here)),source_path=name,sha256=hashlib.sha256(blob).hexdigest(),url='https://github.com/sgl-project/sglang/blob/'+pin+'/'+name))
(out/'manifest.json').write_text(json.dumps(dict(pin=pin,files=records),indent=2))
