"""Finite-coefficient-only experimental WGS source; not enabled by any backend."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
source=here/'hopper_expert_register.cu';s=source.read_text()
old='if ((byte & 127u) == 127u) return serving_pair<R>(word, bytes, table);'
new='if constexpr (!REGISTER_FINITE) {\n    '+old+'\n  }'
assert s.count(old)==1;s=s.replace(old,new)
needle='#ifndef REGISTER_DEQUANT'
s=s.replace(needle,'#ifndef REGISTER_FINITE\n#define REGISTER_FINITE 0\n#endif\n'+needle)
p=here/'hopper_expert_register_finite.cu';p.write_text(s)
report=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),parent_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),not_runtime_enabled=True,required_contract=['scan actual source SF bytes per layer; all(sf&127 !=127)','only immutable prepared layer enters finite kernel','any absent proof, weight replacement/version change or reload uses original table kernel and rebuilds captured graphs','changed input activations remain unrestricted; no new math boundary','inference tensors without version metadata are conservatively unproven unless explicit immutable-owner lifecycle exists'],exceptional_original_fallback=True)
(here/'finite_wgs_source_plan.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
