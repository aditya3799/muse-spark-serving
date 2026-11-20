from pathlib import Path
import copy,json,hashlib
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round2_25x\hopper_moe');O=P.parents[1]/'hopper_moe';D=P/'sm90_corrected';D.mkdir(exist_ok=True);ROOT=P.parents[3]
def rd(p):return p.read_text(encoding='utf-8-sig')
def wr(p,s):p.write_text(s,encoding='utf-8')
for name in ('virtual_k_context.cuh','virtual_k_wgmma.cuh'):
 wr(D/name,rd(O/name))
for name in ('virtual_k_epilogue.cuh','virtual_k_smem_writer.cuh'):
 wr(D/name,rd(P/name))
s=rd(O/'virtual_k_kernel.cuh').replace('BlockShape::M == WarpShape::M','BlockShape::M % WarpShape::M == 0 && BlockShape::M / WarpShape::M <= 2').replace('(BlockShape::N / WarpShape::N) * 32','(BlockShape::M / WarpShape::M) * (BlockShape::N / WarpShape::N) * 32');wr(D/'virtual_k_kernel.cuh',s)
s=rd(O/'virtual_k_codegen.py').replace("'virtual_k_epilogue.cuh')","'virtual_k_epilogue.cuh','virtual_k_smem_writer.cuh')").replace('if bm!=wm or wn!=32','if bm%wm or bm//wm not in (1,2) or wn!=32').replace('physical=(bn//wn)*32','physical=(bm//wm)*(bn//wn)*32');wr(D/'virtual_k_codegen.py',s)
jobs=json.loads(rd(O/'virtual_k_jobs.json'))['jobs']
for j in jobs:j['job_id']+='__writer_fixed'
src=ROOT/'swarm_2026-11-16/moe/phase4_hour/producer_compile_jobs.json';old=next(j for j in json.loads(rd(src))['jobs'] if j['stage']=='w13' and any(lo<32*8<=hi for lo,hi in j['intervals']));conf=copy.deepcopy(old['baseline']);assert conf['block_shape']==[24,256,128] and conf['warp_shape']==[24,32,64],conf
for mode in ('control','virtual','virtual_m2'):
 j=copy.deepcopy(old);j.update(policy=mode,job_id='w13_m32_actual_'+mode,configuration=copy.deepcopy(conf),tokens=32)
 if mode=='virtual_m2':j['configuration']['block_shape'][0]*=2
 jobs.append(j)
wr(D/'virtual_k_jobs.json',json.dumps({'status':'CORRECTED_WRITER_CPU_AUDIT_REQUIRED_OFFLINE_ONLY','source_manifest_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'jobs':jobs},indent=2)+'\n')
s=rd(O/'offline_virtual_k_sm90.py').replace("P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'","P.parents[4]/'swarm_2026-11-16/moe/phase4_hour'").replace('default=5','default=8').replace("assert k.block_shape==baseline.block_shape and k.warp_shape==baseline.warp_shape","assert k.block_shape[1:]==baseline.block_shape[1:] and k.warp_shape==baseline.warp_shape").replace("enabled=policy=='virtual'","enabled=policy.startswith('virtual')")
s=s.replace("row.update(status='compiled',seconds=", "row.update(status='compiled',seconds=")
s=s.replace("  assert not torch.cuda.is_initialized();save();print", "  row['zero_spill_and_stack']=not any(int(v) for v in re.findall(r'(\\d+) bytes (?:spill stores|spill loads|stack frame)',logs))\n  row['resource_qualified']=row['zero_spill_and_stack'] and bool(re.search(r'Used \\d+ registers',logs))\n  assert not torch.cuda.is_initialized();save();print")
wr(D/'offline_virtual_k_sm90.py',s)
expert=P.parent/'expert_partition/SM90_COMPOSED_W13_MANIFEST.json';fjobs=[]
for j in json.loads(rd(expert))['jobs']:
 c={**j['meta'],**j['compute_config'],**j['tuning_config'],'use_cp_async':True};fjobs.append({'stage':'w13','policy':j['policy'],'job_id':j['name'],'intervals':[],'configuration':c,'baseline':copy.deepcopy(c)})
wr(D/'w13_fused_jobs.json',json.dumps({'source_manifest_sha256':hashlib.sha256(expert.read_bytes()).hexdigest(),'jobs':fjobs},indent=2)+'\n')
s=rd(O/'offline_w13_fused_sm90.py').replace("P.parents[2]/'swarm_2026-11-16/moe/phase4_hour'","P.parents[4]/'swarm_2026-11-16/moe/phase4_hour'").replace("P.parent/'coordination'","P.parents[2]/'coordination'").replace("P.parent/'coordination/w13_fused_epilogue.py'","P.parents[2]/'coordination/w13_fused_epilogue.py'").replace('default=2','default=4')
s=s.replace("  assert not torch.cuda.is_initialized();save();print", "  row['zero_spill_and_stack']=not any(int(v) for v in re.findall(r'(\\d+) bytes (?:spill stores|spill loads|stack frame)',logs))\n  row['resource_qualified']=row['zero_spill_and_stack'] and bool(re.search(r'Used \\d+ registers',logs))\n  assert not torch.cuda.is_initialized();save();print")
wr(D/'offline_w13_fused_sm90.py',s)
print('Prepared8 virtual/control and4 composed fused configs; no compiler/GPU activity')
