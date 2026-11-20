"""Exhaustive CPU-only NVFP4 group-scale representation check."""
import json,torch
from pathlib import Path
P=Path(__file__).resolve().parent
values=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)
scales=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).double()
reference=values[:,None]*scales[None,:]
finite=torch.isfinite(reference)
bf16=reference.to(torch.bfloat16);back=bf16.double()
assert torch.equal(reference[finite],back[finite])
assert torch.equal(torch.signbit(reference[finite]),torch.signbit(back[finite]))
fp32=(values.float()[:,None]*scales.float()[None,:]).to(torch.bfloat16)
assert torch.equal(fp32[finite].view(torch.int16),bf16[finite].view(torch.int16))
r={'status':'EXACT_FINITE_REPRESENTATION_PASS','finite_products':int(finite.sum()),'nonfinite_products':int((~finite).sum()),'all_finite_products_exact_in_BF16':True,'finite_signed_zero_preserved':True,'FP32_multiply_then_BF16_matches':True,'scope':'All16 E2M1 codes times all256 E4M3FN group-scale bytes. No global scale folded in. This proves operand representation, not GEMM summation order or serving tokens.','nonfinite':'32 products use E4M3FN NaN scale codes; no NaN payload bitwise claim.'}
(P/'DEQUANT_CPU_CONTRACT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
