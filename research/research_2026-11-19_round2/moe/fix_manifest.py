from pathlib import Path
import json,hashlib
here=Path(__file__).resolve().parent
p=here/'build_dequant_source.py';s=p.read_text(encoding='utf-8-sig');s=s.replace('hashlib.sha256(s.encode()).hexdigest()',"hashlib.sha256((here/'register_dequant_probe.cu').read_bytes()).hexdigest()")
s=s.replace('Actual CUDA intrinsic and NaN lookup tests pending.','Actual CUDA intrinsic and NaN lookup tests passed; see register_dequant_gpu.json.')
p.write_text(s,encoding='utf-8')
p=here/'register_dequant_sources.json';d=json.loads(p.read_text());d['source_sha256']=hashlib.sha256((here/'register_dequant_probe.cu').read_bytes()).hexdigest();d['note']='No FTZ, no FP8 activation or global-scale change. Exhaustive finite, signed-zero, subnormal and NaN payload CUDA proof passed; see register_dequant_gpu.json. Hash is actual file bytes including platform newlines.';p.write_text(json.dumps(d,indent=2))
