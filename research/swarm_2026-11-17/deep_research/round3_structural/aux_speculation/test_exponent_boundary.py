"""CPU arithmetic model of pinned NVFP4 exponent compensation; no GPU proof."""
import torch,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[3];inc=root/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'
q=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64)[:,None]
s=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).double()[None,:];mask=torch.isfinite(s).expand(16,-1)
# Direct transcription get_total/mainloop/epilogue_exp_offset (non-native mixed).
weight_offset=128-2;scale_offset=128-8;total=weight_offset+scale_offset
x=min(127-4+1,total);y=min(total-x,scale_offset);epilogue=total-x-y
assert (total,x,y,epilogue)==(246,124,120,2)
# Model BF16 truncation/rounding at each native multiplication, including intermediate subnormals.
qd=(q*2.**-weight_offset).bfloat16();sd=(s*2.**-scale_offset).bfloat16()
qb=(qd.double()*2.**x).bfloat16();sb=(sd.double()*2.**y).bfloat16()
operand=(qb.double()*sb.double()).bfloat16();direct=(q*s/4).bfloat16()
assert torch.equal(operand[mask].view(torch.int16),direct[mask].view(torch.int16))
# Multiplying BF16 output by4 must stay AFTER FP32 accumulator ->BF16 conversion.
# Demonstrate different underflow boundary under algebraic reassociation.
f=torch.tensor([2.**-135,2.**-134,3*2.**-135],dtype=torch.float64)
original=(f.bfloat16().double()*4).bfloat16();folded=(f*4).bfloat16()
assert not torch.equal(original.view(torch.int16),folded.view(torch.int16))
files=['arith/exp_offset.cuh','arith/mainloop_arith.cuh','arith/epilogue_arith.cuh','mma/wmma.cuh','mma/wgmma.cuh','epilogue/smem_writer.cuh','datatype/dtypes.cuh']
r={'status':'CPU_EXPONENT_BOUNDARY_PASS_NATIVE_PROOF_PENDING','finite_operand_cases':int(mask.sum()),'offsets':{'total':total,'mainloop_weight':x,'mainloop_scale':y,'epilogue':epilogue},'required_mma_operand':'q * group_scale /4 in BF16','required_epilogue':'FP32 accumulator convert toBF16, multiply4 inBF16, unchanged external global postscale. BF16 channel scale4 can express this boundary if same MMA/reduction tree and metadata guards hold.','subnormal_counterexample':{'fp64_accumulator':f.tolist(),'original_after_bf16_then_x4':original.float().tolist(),'reassociated_x4_then_bf16':folded.float().tolist()},'limitations':['CPU port is not native CUDA BF16 instruction proof.','Must keep original MmaOp,BlockK,WarpK,Kbanks,reduction tree; default BF16 tuning is not interchangeable.','No NaN payload proof.','Small operand signedzero tested; accumulator signedzero and overflow need native tests.'],'source_hashes':{name:hashlib.sha256((inc/name).read_bytes()).hexdigest() for name in files}}
(p/'PACKED_EXPONENT_BOUNDARY.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'finite_operand_cases':r['finite_operand_cases']}))
