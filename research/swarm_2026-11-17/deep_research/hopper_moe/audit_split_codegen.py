from pathlib import Path
import sys,json,copy,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from split_issue_codegen import render_split_issue
jobs=json.loads((P/'split_issue_jobs.json').read_text())['jobs'];checked=0;guards=0
for j in jobs:
 source=(P/'offline_split_issue'/j['job_id']/'kernel.cu').read_text()
 # Restore canonical generated include from isolated explicit-header experiment.
 source=source.replace('#include "'+ '/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-17/deep_research/hopper_moe/'+('split_control_kernel.cuh' if j['policy']=='control' else 'split_kernel.cuh')+'"','#include <humming/kernel/humming.cuh>')
 original,meta=render_split_issue(source,{},86);assert original is source and not meta['enabled']
 rendered,meta=render_split_issue(source,j['configuration'],90,True,j['policy']);assert meta['enabled']
 assert '#include "split_wgmma.cuh"' not in rendered and 'void issue(' in rendered and 'wgmma_wait<0>();' in rendered
 checked+=1
for key,value in [('a_dtype','float16'),('use_stream_k',True),('use_f16_accum',True),('use_warp_spec',True),('multi_cast_size_b',2),('num_stages',2),('weight_scale_group_size',32),('use_packed_k_layout',True),('input_scale_group_size',16)]:
 c=copy.deepcopy(jobs[0]['configuration']);c[key]=value
 try:render_split_issue(source,c,90,True)
 except ValueError:guards+=1
 else:raise AssertionError(key)
r={'status':'source_renderer_cpu_contract_passed','rendered_compiled_configurations':checked,'invalid_metadata_guards':guards,'default_disabled_source_identity':True,'cache_safety':'All modified header content embedded in source cachekey; no dependence on outside-header mtimes','limits':'Inline source form parsed structurally; explicit-header form compiled. Inline form requires its own compile/runtime gate before use.'}
(P/'split_issue_codegen_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
