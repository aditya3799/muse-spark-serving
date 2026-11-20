"""Derive a guarded-package fixture; preserve all round-one scripts/receipts."""
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
s=(root/'research_2026-11-19/decode/fa3_prologue_probe.py').read_text()
s=s.replace('from fa3_prologue_kernel import fusion18_kernel,conversion_probe','from prologue_package.kernel import conversion_probe\nfrom prologue_package import Fa3Prologue')
s=s.replace("(HERE/'fa3_prologue_kernel.py')","(HERE/'prologue_package/kernel.py')")
s=s.replace("path=HERE/f'fa3_prologue_{a.seed}.json'","path=HERE/f'component_{a.seed}.json'")
s=s.replace("norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)","norm=SimpleNamespace(_head_dim=64,_num_q_heads=16,_num_kv_heads=2,_eps=1e-5)\n adapter=Fa3Prologue()")
begin=s.index('   def candidate():');end=s.index('   calls=',begin)
s=s[:begin]+'''   def candidate():
    result=adapter.try_run(qkv,positions if sliding else None,rope if sliding else None,
      pools['candidate'][0].view(torch.float8_e4m3fn),pools['candidate'][1].view(torch.float8_e4m3fn),loc,
      reference=lambda x,p,r:native_norm(norm,x,p,r,kv_store=None),reference_is_triton=True,compatible_pool=True)
    assert result is not None,adapter.last_reason
    outputs['candidate']=result.view(torch.uint8)
'''+s[end:]
s=s.replace("save();print(m,sliding,'PASS'","doc['qualification']=adapter.receipts;save();print(m,sliding,'PASS'")
s=s.replace("doc['complete']=True;doc['peak_allocated_bytes']", "doc['complete']=True;doc['qualification']=adapter.receipts;doc['peak_allocated_bytes']")
(here/'component_fixture.py').write_text(s)
print('generated component_fixture.py')
