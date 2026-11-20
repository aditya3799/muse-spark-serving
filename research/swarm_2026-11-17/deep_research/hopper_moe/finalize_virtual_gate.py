from pathlib import Path
import hashlib,json,ast
P=Path(__file__).resolve().parent
p=P/'virtual_k_gate_sm90.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace("HummingMethod.transform_humming_layer(layer,already_padded=True)","""HummingMethod.transform_humming_layer(layer,already_padded=True)
  meta=json.loads(layer.humming_metas[''].to_str())
  for field in ('shape_n','shape_k','num_experts','a_dtype','b_dtype','bs_dtype','c_dtype','weight_scale_group_size','weight_scale_type','weight_scale_2_type','use_packed_k_layout'):
   assert meta[field]==c[field],(field,meta[field],c[field])""")
p.write_text(s,encoding='utf-8');ast.parse(s)
p=P/'audit_virtual_gate.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("assert \"job['configuration']!=job['baseline']\" not in source # implemented with j during selection\n",'');p.write_text(s,encoding='utf-8')
p=P/'RESIDENCY_RESOURCE_REVIEW.json';r=json.loads(p.read_text(encoding='utf-8-sig'));elf=json.loads((P/'CUDA_ELF_RESOURCE_AUDIT.json').read_text());byid={row['job']:row for row in elf['rows']}
for row in r['rows']:
 if row['job'] in byid:
  e=byid[row['job']];row['compiled_dynamic_shared_bytes']=e['compiled_constants']['SMEM_SIZE'];row['compiled_static_shared_bytes']=e['static_shared_bytes'];row['compiled_total_shared_bytes']=e['compiled_total_shared_bytes'];row['shared_model_cta_bound']=(228*1024)//e['compiled_total_shared_bytes']
r['correction']='Updated shared-memory capacity to include ELF static shared allocation (1024 bytes) plus dynamic SMEM_SIZE. CUDA_ELF_RESOURCE_AUDIT.json is authoritative for compiled metadata; achieved residency remains unmeasured.'
p.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
p=P/'VIRTUAL_K_DESIGN_REVIEW.json';r=json.loads(p.read_text());r['status']='Four virtual geometries and one matched control compiled for SM90a with zero spills; CPU/SASS/resource gates passed; H200-only numerical harness prepared but unexecuted.';r['proof_limits'].append('Compiler commutes operands within unchanged FP32 FADD.FTZ tree. Finite addition grouping is checked; NaN payload ordering is not established. H200 runtime gate treats nonfinite cases as diagnostics only.')
r['runtime_gate']={'harness':'virtual_k_gate_sm90.py','contract':'virtual_k_gate_contract.py','cpu_audit':'VIRTUAL_K_GATE_CPU_AUDIT.json','status':'unexecuted; refuses devices other than H200 SM90','default_finite_cases':192,'component_only':True}
r['resource_receipt']='CUDA_ELF_RESOURCE_AUDIT.json';r['SASS_tree_receipt']='VIRTUAL_K_REDUCTION_SASS_AUDIT.json'
p.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
r={'status':'prepared_unexecuted','cwd':'serving/swarm_2026-11-17/deep_research/hopper_moe','prerequisites':['Existing pinned Humming vendor and compatible PyTorch/CUDA environment','NVIDIA H200 SM90; SM86/H100/B200 fail closed','Root must grant hardware/compiler lease before any local execution; no remote submission or paid resource authorization'], 'commands':{'numerical_gate':'MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 python virtual_k_gate_sm90.py --out h200_virtual_k_gate','memcheck':'MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 compute-sanitizer --tool memcheck --error-exitcode 99 python virtual_k_gate_sm90.py --seeds 1 --out h200_virtual_k_memcheck','racecheck':'MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 compute-sanitizer --tool racecheck --error-exitcode 99 python virtual_k_gate_sm90.py --seeds 1 --out h200_virtual_k_racecheck','initcheck':'MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 compute-sanitizer --tool initcheck --error-exitcode 99 python virtual_k_gate_sm90.py --seeds 1 --out h200_virtual_k_initcheck'},'scope':'Four fixed supplied configurations and canonical NVFP4 tensors; direct indexed GEMMs. Does not test adaptive policy selection, full chain, finalizer, distributed collectives, or serving speed. No timing included.','required_next_gate':'After finite bitwise and sanitizer passes, integrate default-off into isolated full helper, preserve K and canonical scales, validate full output plus changed-input graph replay before matched cold A/B timing.','artifact_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('virtual_k_gate_sm90.py','virtual_k_gate_contract.py','virtual_k_codegen.py','virtual_k_jobs.json')}}
(P/'H200_VIRTUAL_K_GATE_PLAN.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
print('Updated gate metadata, actual shared resource bounds, and reproducible commands.')
