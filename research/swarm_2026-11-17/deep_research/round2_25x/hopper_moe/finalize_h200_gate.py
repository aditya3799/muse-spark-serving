from pathlib import Path
import json,hashlib,ast,copy,sys
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mf=P/'H200_CORRECTED_GATE_MANIFEST.json';r=json.loads(mf.read_text());r['pinned_vendor_hashes']={str(f.relative_to(V)).replace('\\','/'):sha(f) for f in sorted(list((V/'humming').rglob('*.py'))+list((V/'humming/include').rglob('*.cuh')))};mf.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
f=P/'h200_corrected_contract.py';s=f.read_text(encoding='utf-8-sig');s=s.replace("P=Path(__file__).resolve().parent","P=Path(__file__).resolve().parent\nEXPECTED_MANIFEST_SHA256="+repr(sha(mf)))
s=s.replace("manifest=json.loads(path.read_text(encoding='utf-8-sig'))", "manifest=json.loads(path.read_text(encoding='utf-8-sig'))\n if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_MANIFEST_SHA256:raise ValueError('Gate manifest changed')\n V=P.parents[3]/'swarm_2026-11-16/moe/humming_vendor'\n for name,digest in manifest['pinned_vendor_hashes'].items():\n  if hashlib.sha256((V/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned vendor changed: '+name)")
f.write_text(s,encoding='utf-8');ast.parse(s);ast.parse((P/'h200_corrected_gate.py').read_text(encoding='utf-8-sig'))
sys.path.insert(0,str(P));from h200_corrected_contract import *
m=load_manifest();passed=0
for major,minor,name,valid in [(9,0,'NVIDIA H200',True),(8,6,'RTX3050',False),(9,0,'H100',False),(10,0,'H200',False)]:
 try:require_h200(major,minor,name)
 except RuntimeError:assert not valid
 else:assert valid
 passed+=1
for pair in m['pairs']:
 for side in ('baseline','candidate'):
  c=pair[side]['configuration'];validate_config(c)
  for key,value in [('use_tma_b',True),('use_warp_spec',True),('use_f16_accum',True),('num_stages',2),('input_scale_group_size',16),('a_dtype','float8e4m3'),('num_write_splits',2),('use_stream_k',True)]:
   bad=copy.deepcopy(c);bad[key]=value
   try:validate_config(bad)
   except ValueError:passed+=1
   else:raise AssertionError(key)
  for pattern in ('uniform','one_hot','eight_hot','mixed','all_remote'):
   ids=local_routes(pair['tokens'],c['num_experts'],pattern,5);si,ei,np=aligned_routes(ids,c['num_experts'],c['block_shape'][0]);active=[i for i,v in enumerate(ids) if v>=0];got=[i for i in si[:np] if i<len(ids)];assert sorted(got)==active and len(got)==len(set(got));assert all(ei[j//c['block_shape'][0]]==ids[idx] for j,idx in enumerate(si[:np]) if idx<len(ids));passed+=1
r={'status':'CPU_PREFLIGHT_PASS_NOT_H200_EXECUTION','pairs':len(m['pairs']),'contract_checks':passed,'manifest_sha256':sha(mf),'gate_sha256':sha(P/'h200_corrected_gate.py'),'contract_sha256':sha(f),'native_runtime_executed':False,'old_gate_status':'Unchanged HOLD_WRITER_OWNERSHIP_BUG','commands':['MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 python h200_corrected_gate.py --out FRESH_VIRTUAL_OUTPUT --family virtual','MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 python h200_corrected_gate.py --out FRESH_WAIT1_OUTPUT --family wait1','MAX_JOBS=1 HUMMING_DISABLE_PARALLEL_BUILD=1 python h200_corrected_gate.py --out FRESH_WM48_OUTPUT --family wm48'],'limits':['Finite direct indexed GEMM only; no full-helper/serving or timing qualification.','Wait1 static source/SASS audit is not native racecheck; run relevant sanitizer before any promotion.','WM48 changes native opcode width, so this gate falsifies equivalence instead of presuming it.']}
(P/'H200_CORRECTED_GATE_CPU_AUDIT.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
