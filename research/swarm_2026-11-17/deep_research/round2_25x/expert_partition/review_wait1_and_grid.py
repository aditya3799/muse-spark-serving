import ast,copy,difflib,hashlib,importlib.util,json,pathlib,re
p=pathlib.Path(__file__).resolve().parent;root=p.parents[3];other=p.parent/'hopper_moe';vendor=root/'swarm_2026-11-16/moe/humming_vendor/humming/include/humming';snap=p/'wait1_review_snapshot';snap.mkdir(exist_ok=True);sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
paths=[other/'wait1'/name for name in ('wait1_codegen.py','wait1_wgmma.cuh','wait1_kernel.cuh')]+[other/'model_wait1_lifetimes.py',other/'WAIT1_LIFETIME_MODEL.json']
hashes={}
for f in paths:(snap/f.name).write_bytes(f.read_bytes());hashes[str(f.relative_to(root))]=sha(f)
for f in (vendor/'mma/wgmma.cuh',vendor/'kernel/humming.cuh',vendor/'memory/s2r_pipeline.cuh',vendor/'memory/g2s_pipeline.cuh',vendor/'arith/mainloop_arith.cuh',vendor/'utils/storage.cuh',vendor/'utils/ptx/wgmma.cuh'):hashes[str(f.relative_to(root))]=sha(f)
spec=importlib.util.spec_from_file_location('wait1_cpu_review',snap/'wait1_codegen.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
jobs=json.loads((other/'sm90_corrected/virtual_k_jobs.json').read_text())['jobs'];base=next(x['configuration'] for x in jobs if x['job_id']=='w13_m32_actual_control');dummy='#include <humming/kernel/humming.cuh>'
positive=[]
for stages in (3,4):
 for tiles in (1,2,3,4,5,7,8,9,31,32,33,63,64,65):
  c=copy.deepcopy(base);c['num_stages']=stages;c['shape_k']=tiles*c['block_shape'][2];rendered,_=mod.render_wait1(dummy,c,90,True);assert 'mma.drain_async();' in rendered;positive.append((stages,tiles))
mutations={'num_stages':2,'shape_k':4097,'input_scale_group_size':16,'use_tma_b':True,'use_warp_spec':True,'use_stream_k':True,'use_f16_accum':True,'use_packed_k_layout':True,'weight_scale_group_size':32,'a_dtype':'float16','b_dtype':'bfloat16','num_write_splits':2,'multi_cast_size_a':2,'use_cp_async':False,'has_bias':True};negative=[]
for key,val in mutations.items():
 c=copy.deepcopy(base);c[key]=val
 try:mod.render_wait1(dummy,c,90,True)
 except ValueError:negative.append({'field':key,'value':val,'rejected':True})
 else:raise AssertionError(('admitted unsupported',key))
for arch in (86,100):
 try:mod.render_wait1(dummy,base,arch,True)
 except ValueError:negative.append({'architecture':arch,'rejected':True})
 else:raise AssertionError(arch)
original=(vendor/'mma/wgmma.cuh').read_text();candidate=(snap/'wait1_wgmma.cuh').read_text(encoding='utf-8-sig');drain='''  CUDA_INLINE void drain_async() {
    wgmma_wait<0>();
    // Prevent compiler movement of accumulator consumption above the final wait.
    fence_regs(regs_c[0]);
  }

'''
normalized=candidate.replace('research_wait1_make_smem_desc','make_wgmma_smem_desc').replace('ResearchWait1WGMMA','WGMMA').replace('wgmma_wait<1>();','wgmma_wait<0>();').replace(drain,'')
assert normalized==original,'Unexpected WGMMA arithmetic diff'
report={'status':'CPU_SOURCE_LIFETIME_REVIEW_PASS_PENDING_COMPILER_AND_DEVICE_GATES','source_hashes':hashes,'positive_renderer_cases':len(positive),'negative_renderer_cases':negative,'exact_normalized_WGMMA_source_equality':True,'independent_model':'WAIT1_INDEPENDENT_MULTIWG_MODEL.json','findings':['Native MMA/dequant/K partition expressions remain unchanged after removing explicit wait/drain/naming edits.','After commit+wait1 only current group may remain pending. Alternate-bank transform overwrites the completed preceding bank; packed QB and scale loads use separate arrays.','ElementA16 makes accumulator scale postprocessing return immediately and final_regs_c_index0 makes per-run may_fence_regs a no-op; otherwise wait1 would permit illegal scalar C access.','Shared-A producer overwrite targets prior ring stage. Current-stage iter0 drains prior-stage last group, and canonical penultimate-iteration CTA barrier joins this guarantee across every warpgroup.','Final wait0 occurs before consumer channel read, shared-reduction epilogue and next scheduler-block zero_accum. Stage and reduce storage alias, so drain is necessary for both input and accumulator lifetime.','Explicit final fence_regs(regs_c[0]) adds compiler register dependencies after wait0; memory clobber alone should not be used as a register-lifetime proof.','Uneven ring tails and one-K-tile cases retain final drain; nonmultiple BlockK element tails are rejected.','Default-off renderer leaves original source unchanged when disabled.'],'limits':['CPU source and abstract ordering proof only. No generated SASS audit yet.','Compiler may coalesce operand registers or insert synchronization; verify realized B-bank lifetimes, waits, spills and lack of early C reads in SASS.','No H200 numerical or racecheck qualification; no latency/overlap gain inferred.','Original cp.async load readiness protocol retained, not independently reimplemented.'],'primary_sources':[{'url':'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html','finding':'wait_group N completes older groups while at most N newest remain pending; asynchronous input registers and accumulators cannot be reused before appropriate completion.'}]}
(p/'WAIT1_SOURCE_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
# Independently verify native policy/task model against pinned heuristic table and actual compile receipts.
model=json.loads((other/'NATIVE_M32_WORK_TILE_MODEL.json').read_text());table=json.loads((root/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json').read_text())['stages']['w13'];heavy=next(c for lo,hi,c in table['intervals'] if lo<256<=hi);light=next(c for lo,hi,c in table['intervals'] if lo<32<=hi);assert heavy['block_shape']==[24,256,128] and light['block_shape'][0]==8
compile_path=other/'offline_sm90_01/virtual/results.json';compiled=json.loads(compile_path.read_text())['cases'];resources=[]
for name in ('w13_m32_actual_control','w13_m32_actual_virtual_m2'):
 item=next(c for c in compiled if c['job_id']==name);regs=int(re.search(r'Used (\d+) registers',' '.join(item['resource_lines'])).group(1));threads=item['resolved']['total_threads'];assert 2*regs*threads>65536
 resources.append({'job':name,'registers':regs,'threads':threads,'shared_bytes_estimated':item['shared_bytes_estimated'],'zero_spill':item['zero_spill_and_stack'],'unrounded_registers_per_CTA':regs*threads,'register_capacity_upper_bound_CTAs_per_SM':65536//(regs*threads)})
rows=[]
for item in model['rows']:
 counts=item['heavy_counts_cutoff8'];old=sum((c+23)//24*32 for c in counts);new=sum(((c+47)//48 if (c+47)//48<(c+23)//24 else (c+23)//24)*32 for c in counts);assert old==item['original_heavy_logical_work_tiles'] and new==item['selective_m2_logical_work_tiles']
 rows.append({'pattern':item['pattern'],'baseline_tasks':old,'candidate_tasks':new,'nonempty_persistent_CTAs_for_grid132':[min(132,old),min(132,new)],'max_tasks_per_CTA_for_grid132':[(old+131)//132,(new+131)//132],'notes':'Logical scheduler task count, not measured simultaneous SM residency. Mixed retains BM24 fallback.'})
grid={'status':'NATIVE_M32_GRID_MODEL_INDEPENDENT_CPU_REVIEW_PASS','source_sha256':{str(f.relative_to(root)):sha(f) for f in (other/'NATIVE_M32_WORK_TILE_MODEL.json',compile_path,vendor/'scheduler.cuh')},'actual_policies':{'cutoff':8,'light':light,'heavy':heavy,'M2_BM':48},'rows':rows,'resources':resources,'findings':['Persistent launch grid defaults num_sms*num_ctas_per_sm with num_ctas_per_sm1. Source scheduler distributes MN tasks cyclically across fixed CTAs; gridDim does not shrink with active tasks.','One legal hot expert gives64 baseline tasks versus32 M2 tasks, both below assumed132 H200 SM slots. Both kernels have512 threads; resource upper bound is one CTA per SM, before allocation rounding or other limits.','Both one-hot policies compute48 padded expert rows total:2*BM24 vs1*BM48. Weight traversals halve, but each nonempty candidate task covers twice the padded rows and available independent tasks halve.','Eight hot experts give512 versus256 tasks. Mixed heavy counts24 get no native BM24→48 tile-count benefit and retain original heavy path.','SM86 gain cannot transfer directly: cutoff16/BM16→32 differs from native SM90 cutoff8/BM24→48, and local20SM versus132SM changes underfill.'],'limits':['132SM is supplied H200 target assumption, not live device query','Register upper bound uses64K32bit Hopper registers; actual occupancy/placement needs native CUDA occupancy/runtime evidence','Compile-only resources do not prove arithmetic, hazard safety or performance'],'primary_source':'https://docs.nvidia.com/cuda/hopper-tuning-guide/'}
(p/'NATIVE_M32_GRID_CPU_REVIEW.json').write_text(json.dumps(grid,indent=2)+'\n')
risk=json.loads((p/'UNIFIED_COMPOSITION_RISKS.json').read_text(encoding='utf-8-sig'));risk['native_heavy_resource_attribution']={'finding':'Canonical and fused heavy BM176/BM160 controls have matching255-register and spill/stack reports. No added fusion pressure is established by those matching counts.','scope':'Root paired control attribution; see hopper_moe/offline_sm90_01/fused/results.json','memory':'If an unfused-heavy fallback is chosen, canonical raw2I scratch is needed and all-fused2R*H savings no longer apply; this conditional memory statement does not itself require rejecting fused-heavy.'};(p/'UNIFIED_COMPOSITION_RISKS.json').write_text(json.dumps(risk,indent=2)+'\n')
print(json.dumps({'wait1_positive':len(positive),'wait1_negative':len(negative),'native_grid_rows':len(rows),'resources':resources}))

