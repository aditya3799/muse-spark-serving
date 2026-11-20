"""Prepare bounded validation of an isolated BLOCK32 eight-warp launch policy."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
s=(H/'prefill_prologue_bytes.py').read_text(encoding='utf-8-sig')
s=s.replace('import torch\n','import torch,triton\nfrom prologue_package.kernel import fusion18_kernel\n')
anchor='@torch.no_grad()\ndef main():'
launch='''class Warp8Prologue(Fa3Prologue):
    # Diagnostic launch policy only. Frozen source/runtime is not modified.
    def _launch(self,qkv,positions,rope,k,v,locations,eps,codec):
        m=qkv.shape[0];block=self._block(m)
        out=torch.empty((m,1024),device=qkv.device,dtype=torch.uint8)
        fusion18_kernel[(triton.cdiv(m*18,block),)](
            qkv,out,qkv,positions if positions is not None else qkv,rope if rope is not None else qkv,
            k.view(torch.uint8),v.view(torch.uint8),locations,float(eps),qkv.stride(0),m*18,k.shape[0],
            16,2,64,64,block,positions is not None,True,True,False,NATIVE=codec=='native',
            num_warps=8 if block==32 else 4,debug=True)
        return out.view(torch.float8_e4m3fn)

'''
assert s.count(anchor)==1
s=s.replace(anchor,launch+anchor).replace('adapter=Fa3Prologue()','adapter=Warp8Prologue()')
s=s.replace('(16383,16384)]:','(16383,16384),(16384,16384)]:')
s=s.replace("(HERE/'prefill_prologue_bytes.json')","(HERE/'prologue_warp8/byte_matrix.json')")
old='    assert all(checks.values()),records[-1]'
new='''    if not all(checks.values()):
     failure=dict(scope=__doc__,passed=False,records=records,first_failure=records[-1],
       policy='8warps only BLOCK32; otherwise frozen4warps',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
     (HERE/'prologue_warp8/byte_matrix.json').write_text(json.dumps(failure,indent=2))
     raise AssertionError(records[-1])'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("production_shape_gate_bypassed_only_in_fixture=True,", "policy='8warps only BLOCK32; otherwise frozen4warps',production_shape_gate_bypassed_only_in_fixture=True,")
ast.parse(s);(H/'prologue_warp8_matrix.py').write_text(s)
s=(H/'compile_prologue_warp8.py').read_text(encoding='utf-8-sig')
s=s.replace('HAS_ROPE=False','HAS_ROPE=True').replace('sm90_full_nativefp8','sm90_rope_nativefp8').replace('offline_receipt.json','offline_rope_receipt.json').replace('full-attention layout probe','RoPE layout probe').replace('One full-attention specialization only; no RoPE or H200 execution','One RoPE specialization only; no H200 execution')
ast.parse(s);(H/'compile_prologue_warp8_rope.py').write_text(s)
print('prepared42 changed-input matrix cells plus metadata identity controls; one offlineRoPE specialization')
