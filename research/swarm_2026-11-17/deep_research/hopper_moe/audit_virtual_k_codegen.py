from pathlib import Path
import sys,json,copy,ast
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));from virtual_k_codegen import render_virtual_k
jobs=json.loads((P/'virtual_k_jobs.json').read_text())['jobs'];checks=[]
for job in jobs:
 original=P/'offline_split_issue'/('%s_m%d_control'%(job['stage'],job['tokens']))/'kernel.cu'
 source=original.read_text();source=source.replace('#include "/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-17/deep_research/hopper_moe/split_control_kernel.cuh"','#include <humming/kernel/humming.cuh>')
 assert render_virtual_k(source,{},86)[0] is source
 if job['policy']=='control':continue
 rendered,manifest=render_virtual_k(source,job['configuration'],90,True)
 assert rendered.count('void reduce_virtual_partitions()')==1 and 'virtual_part * BlockShape::M * kSwizzleBytes' in rendered
 assert 'if constexpr (BlockShape::K > WarpShape::K) smem_reducer.reduce(regs_c_ptr);' not in rendered
 assert 'regs_c[virtual_part][delta_j + j][0], scale_d' in rendered
 assert '#include "virtual_k_' not in rendered
 checks.append({'job':job['job_id'],'manifest':manifest})
guards=0
for key,value in [('a_dtype','float16'),('use_f16_accum',True),('use_warp_spec',True),('use_stream_k',True),('num_stages',2),('multi_cast_size_b',2),('input_scale_group_size',16),('num_write_splits',2)]:
 c=copy.deepcopy(jobs[0]['configuration']);c[key]=value
 try:render_virtual_k(source,c,90,True)
 except ValueError:guards+=1
 else:raise AssertionError(key)
for f in ('offline_virtual_k_sm90.py','offline_small_k_repair_sm90.py'):ast.parse((P/f).read_text(encoding='utf-8-sig'))
r={'status':'CPU_source_contract_passed','checks':checks,'invalid_guards':guards,'limits':'Generated source structurally audited; not compiled or executed.'}
(P/'virtual_k_source_contract.json').write_text(json.dumps(r,indent=2)+'\n');print('Virtual sourcecontracts',len(checks),'guards',guards)
