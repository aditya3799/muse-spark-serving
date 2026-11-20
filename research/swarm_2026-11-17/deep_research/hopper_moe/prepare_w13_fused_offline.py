from pathlib import Path
import ast,hashlib,json
P=Path(__file__).resolve().parent;C=P.parent/'coordination';renderer=C/'w13_fused_epilogue.py'
source=renderer.read_text(encoding='utf-8-sig');tree=ast.parse(source)
ns={'INC':P.parents[2]/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming'}
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('replace_once','render')]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(renderer),'exec'),ns)
original=(P/'offline_virtual_k/w13_m8_s4_virtual/kernel.cu').read_text()
# Use canonical template body from an earlier source, not an already-injected virtual kernel.
original=(P/'offline_split_issue/w13_m8_control/kernel.cu').read_text().replace('#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-17/deep_research/hopper_moe/split_control_kernel.cuh"','#include <humming/kernel/humming.cuh>')
rendered=ns['render'](original)
assert rendered.count('const __nv_bfloat16 *post_scales) {')==1
assert 'uint2 *gmem_ptr_raw;' in rendered and 'uint2 *gmem_ptr;' in rendered
assert 'int4 val = ctx.smem.reduce[smem_offset_swizzled];' in rendered
assert 'reinterpret_cast<__nv_bfloat16 *>(&val)' in rendered
assert 'gmem_writer.set_expert(expert_id);' in rendered
assert 'ctx.params.bs2' in rendered # Original scale slot semantics preserved, not repurposed.
checks=0
for n in (256,8192):
 for bn in (128,256):
  for row in (0,1,17,2047):
   for tile in range(n//bn):
    for col in range(bn//8):
     byte_offset=(row*(n//8)+tile*(bn//8)+col)*8
     assert byte_offset==row*n+(tile*bn+col*8)
     assert row*n<=byte_offset and byte_offset+8<=(row+1)*n
     for pair in range(4):
      original_gate_column=tile*bn+col*8+pair*2
      fused_column=tile*(bn//2)+col*4+pair
      assert fused_column==original_gate_column//2
     checks+=1
jobs0=json.loads((P/'split_issue_jobs.json').read_text())['jobs'];jobs=[]
for tokens in (8,256):
 j=next(j for j in jobs0 if j['stage']=='w13' and j['tokens']==tokens and j['policy']=='control')
 jobs.append({'job_id':'w13_m%d_fused'%tokens,'stage':'w13','tokens':tokens,'policy':'fused','configuration':j['configuration'],'baseline':j['configuration'],'intervals':j['intervals']})
manifest={'scope':'Offline SM90 fused W13 legacy epilogue. No runtime guard changes or device execution; original K geometry retained, output interleave prerequisite external.','renderer_sha256':hashlib.sha256(renderer.read_bytes()).hexdigest(),'jobs':jobs}
(P/'w13_fused_jobs.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
s=(P/'offline_occupancy_sm90.py').read_text().replace("P/'occupancy_jobs.json'","P/'w13_fused_jobs.json'").replace('default=12','default=2')
s=s.replace('from producer_tuning import producer_config',"sys.path.insert(0,str(P.parent/'coordination'))\nfrom w13_fused_epilogue import render")
s=s.replace("files=[OLD/'producer_tuning.py'","files=[P.parent/'coordination/w13_fused_epilogue.py'")
s=s.replace("  row['shared_bytes_estimated']=k.estimated_smem_size", "  generated=render(k.code)\n  assert not k.use_tma and not k.use_warp_spec and k.pad_shape[1:]==(0,0)\n  row['shared_bytes_estimated']=k.estimated_smem_size")
s=s.replace("(dest/'kernel.cu').write_text(k.code)","(dest/'kernel.cu').write_text(generated)").replace('hashlib.sha256(k.code.encode())','hashlib.sha256(generated.encode())').replace('Compiler.compile(k.code,','Compiler.compile(generated,')
s=s.replace("assert not k.use_tma and not k.use_warp_spec and k.pad_shape[1:]==(0,0)","assert not k.use_tma and not k.use_warp_spec and all(x==0 for x in k.pad_shape[1:])")
(P/'offline_w13_fused_sm90.py').write_text(s,encoding='utf-8');ast.parse(s)
r={'status':'CPU_source_address_review_passed','renderer_sha256':hashlib.sha256(renderer.read_bytes()).hexdigest(),'vector_stores_checked':checks,'findings':['Contraction from int4 to uint2 pointers consistently halves physical output row and tile strides while preserving index equations.','Shared int4 BF16 loads and four pairs are preserved; explicit FP32 multiply and BF16 rounding maintain the separate scaling boundary in source.','Post scale comes from an explicit 18th kernel argument, not the bs2 slot. Existing indexed scheduler only enters epilogue.seek for scheduled padded blocks; valid expert IDs remain a caller contract.','Original K reducer and mainloop remain unchanged; expf/fast-math numerical equivalence still requires H200 bitwise tests.'], 'runtime_contract_gaps':['Root has now added int32, contiguity, and device guards following this review; these preconditions are explicit.','run() does not establish that W13 weight and scale rows were interleaved together before canonical repack; the dedicated preparation probe provides that invariant.'],'compile_plan':'W13 M8 original two-way K partition and W13 M256 single K partition/high-register control; no compiler launch yet.'}
(P/'W13_FUSED_INDEPENDENT_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
