from pathlib import Path
import json,copy
P=Path(r'D:\OpenKernelChallenge\serving\swarm_2026-11-17\deep_research\round3_structural\hopper_moe');ROOT=P.parents[3];src=ROOT/'swarm_2026-11-16/moe/phase4_hour/producer_compile_jobs.json';old=json.loads(src.read_text())['jobs'];jobs=[]
for m in (8,32,768):
 for stage in ('w13','w2'):
  orig=next(j for j in old if j['stage']==stage and any(lo<m*8<=hi for lo,hi in j['intervals']));base=copy.deepcopy(orig['baseline']);c=copy.deepcopy(base);c.update(gemm_type='grouped_contiguous',use_warp_spec=True,use_tma=True,use_mbarrier=True,use_cp_async=True,use_tma_a=True,use_tma_b=True,use_tma_bs=True,use_tma_c=True,use_tma_as=False,use_tma_bzp=False,use_tma_bias=False,use_tma_bs2=False,multi_cast_size_a=1,multi_cast_size_b=1)
  jobs.append({'job_id':stage+'_m'+str(m)+'_grouped_tma_ws','stage':stage,'policy':'grouped_tma_ws','tokens':m,'configuration':c,'baseline':base,'intervals':orig['intervals']})
(P/'SM90_COMPACT_JOBS.json').write_text(json.dumps({'scope':'Six architectural grouped-contiguous TMA A/B/scales/C+WS builds. Same native geometry/Kbanks as original indexed; no streamK or lowerprecision. No H200 runtime implied.','jobs':jobs},indent=2)+'\n',encoding='utf-8')
r2=P.parents[1]/'round2_25x/hopper_moe/sm90_corrected/offline_virtual_k_sm90.py';s=r2.read_text();s=s.replace("P.parents[4]/'swarm_2026-11-16/moe/phase4_hour'","P.parents[3]/'swarm_2026-11-16/moe/phase4_hour'").replace('default=8','default=6').replace('from virtual_k_codegen import render_virtual_k','from compact_codegen import render_compact').replace("P/'virtual_k_codegen.py'","P/'compact_codegen.py'").replace("*sorted(P.glob('virtual_k_*.cuh')),",'').replace("P/'virtual_k_jobs.json'","P/'SM90_COMPACT_JOBS.json'")
s=s.replace('assert k.mma_type==MmaType.WGMMA and k.is_indexed_gemm and not k.use_tma_a and not k.use_tma_c and not k.use_tma_as','assert k.mma_type==MmaType.WGMMA and k.is_grouped_contiguous_gemm and k.use_tma_a and k.use_tma_b and k.use_tma_c and k.use_warp_spec and not k.use_tma_as')
s=s.replace("render_virtual_k(k.code,configuration,90,enabled=policy.startswith('virtual'))","render_compact(k.code,configuration,enabled=True)")
(P/'offline_compact_sm90.py').write_text(s,encoding='utf-8');print('Prepared6native architectural jobs; no compile')
