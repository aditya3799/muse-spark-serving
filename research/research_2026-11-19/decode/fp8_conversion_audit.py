"""CPU-only installed FP8 conversion audit; does not imply Hopper parity."""
import hashlib,json
from pathlib import Path
import torch

here=Path(__file__).resolve().parent
header=Path(torch.__file__).parent/'include/torch/headeronly/util/Float8_e4m3fn.h'
values=[-float('inf'),-512.,-480.,-465.,-464.,-448.,-0.,0.,448.,464.,465.,479.,480.,512.,float('inf'),float('nan')]
x=torch.tensor(values)
fp8=x.to(torch.float8_e4m3fn)
bf=x.to(torch.bfloat16).to(torch.float8_e4m3fn)
doc=dict(scope=__doc__,torch_version=torch.__version__,torch_git=torch.version.git_version,
 input=values,cpu_fp32_codes=fp8.view(torch.uint8).tolist(),cpu_bf16_codes=bf.view(torch.uint8).tolist(),
 header=str(header),header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
 cuda_tested=False,warning='Installed headers and wheel runtime are separate evidence; verify exact deployed CUDA behavior before matching exceptional-value semantics.')
(here/'fp8_conversion_audit.json').write_text(json.dumps(doc,indent=2))
print(json.dumps(doc,indent=2))
