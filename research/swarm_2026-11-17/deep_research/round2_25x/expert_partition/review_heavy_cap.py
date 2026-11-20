import json,hashlib,copy,ast
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[3]; H=P.parent/'hopper_moe'; V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,d): (P/name).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def ceildiv(x,y): return (x+y-1)//y
def intervals(values):
 out=[]
 for c in values:
  if out and c==out[-1][1]+1: out[-1][1]=c
  else: out.append([c,c])
 return out
manifest=read(P/'SM90_COMPOSED_W13_MANIFEST.json'); native=[j for j in manifest['jobs'] if j['policy']=='heavy']
sources=[V/'config/mma.py',V/'kernel/repack_weight.py',V/'transform.py',V/'include/humming/mma/wgmma.cuh',V/'include/humming/utils/storage.cuh',P/'SM90_COMPOSED_W13_MANIFEST.json',H/'offline_sm90_01/fused/results.json',H/'offline_heavy_controls_01/results.json']
rows=[]; jobs=[]; assertions=0
for baseline in native:
 old=baseline['tuning_config']['block_shape'][0]
 for bm in [128,136,144,152]:
  equal=[]; more=[]; lesspad=[]; equalpad=[]; morepad=[]
  for c in range(17,16385):
   a,b=ceildiv(c,old),ceildiv(c,bm)
   assert b>=a and a*old>=c and b*bm>=c; assertions+=3
   (equal if a==b else more).append(c)
   (lesspad if b*bm<a*old else equalpad if b*bm==a*old else morepad).append(c)
  sample=[]
  for c in sorted(set([17,48,49,64,65,128,129,136,137,144,145,152,153,160,161,176,177,256,384,512,768,1024,2048,3012,4096,8192,16384])):
   a,b=ceildiv(c,old),ceildiv(c,bm)
   sample.append({'count':c,'old_mtiles':a,'candidate_mtiles':b,'old_padded_rows':a*old,'candidate_padded_rows':b*bm,'logical_weight_requested_ratio':b/a,'logical_work_tasks_old':a*32,'logical_work_tasks_candidate':b*32})
  row={'tokens_policy':baseline['tokens'],'baseline_bm':old,'candidate_bm':bm,'count_domain':[17,16384],'domain_count':16368,'same_weight_tile_count':len(equal),'higher_weight_tile_count':len(more),'padding_lower_equal_higher':[len(lesspad),len(equalpad),len(morepad)],'same_weight_tile_intervals':intervals(equal),'first_extra_weight_tile_interval':[bm+1,old],'worst_weight_tile_ratio':max(ceildiv(c,bm)/ceildiv(c,old) for c in range(17,16385)),'asymptotic_weight_tile_ratio':old/bm,'samples':sample}
  rows.append(row)
for bm in [128,136,144,152]:
 job=copy.deepcopy(native[0]);job['name']='w13_heavy_bm'+str(bm);job['policy']='research_heavy_cap';job.pop('tokens');job['tuning_config']['block_shape'][0]=bm;job['tuning_config']['warp_shape'][0]=bm
 job['eligibility']='Offline compile proposal only; native numerical and timing gates required. No renderer eligibility bypass in serving.'
 job['native_ptx']='wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16'%bm
 job['active_accumulator_values_per_thread']=bm
 job['staged_a_bytes']=4*bm*64*2
 job['stage_b_and_scale_bytes']=4*(256*64//2+256*4)
 job['shared_bytes_source_estimate']=ceildiv(max(job['staged_a_bytes']+job['stage_b_and_scale_bytes'],bm*256*2)+8*bm,1024)*1024
 jobs.append(job)
report={'status':'CPU_ONLY_GEOMETRY_HYPOTHESIS_NO_COMPILE_NO_NATIVE_EXECUTION','source_sha256':{str(p):sha(p) for p in sources},'arithmetic_contract':['Preserve BF16 inputs, NVFP4 weight decoding/FP8 GS16 scale, FP32 accumulator, BF16 output and original postscale/SiLU cast boundaries.','Preserve BK64/WK64, logical N256/WN32, four pipeline stages, no StreamK, no K-partition or reduction topology change.','BM=WM changes native WGMMA N dimension, shared descriptors and output layout. Same source K sequence is NOT native bitwise proof; exact H200 raw and final comparisons required.','No change to interleaved-vs-canonical weight row semantics. Use matching activation for interleaved representation.'], 'weight_layout_contract':['RepackWeightKernel parameters and transform_humming_weight call contain dtype/layout/N/K and not tuning BM/WM. All proposed configurations reuse the same packed weight and scale tensors.','NVFP4+GS16 FP8 logical weight+scale bytes per expert full-K traversal: N*K/2 + N*K/16 = 18874368 bytes for N8192 K4096. Smaller BM never reduces count of complete traversals; it can reduce padding/accumulators and increase parallel tasks.','Requested logical bytes are not measured HBM bytes; repeated tiles may hit caches. Metadata, activation reads, spill traffic and scheduling are separate.'], 'resource_hypothesis':['For MmaShape(BM,64,16), each native instruction has BM/2 FP32 accumulators/thread. WN32 produces two N64 subtiles, one live C bank totals BM values/thread. Declared second bank is not assumed physically live.','256 threads unchanged. Staged A footprint decreases512 bytes per BM row over4stages; B/scale stage footprint unchanged. Shared-size estimates are not compiler/resource-query evidence.','Original BM160/176 canonical and fused controls already have equal255-register spill reports. Any cap can help either; no fusion-specific spill attribution.','BM128 and144 first bracket an informative reduction. Compiler register allocation can be discontinuous; no claim of spill-free threshold until measured.'], 'counts_scope':'Exhaustive integer counts17..16384; each count legal when token population>=count with unique top8. Counts below actual cutoff48/64 would remain light. Equal-count enumeration is NOT a router probability distribution.', 'assertions':assertions,'comparisons':rows,'candidate_jobs':jobs,'primary_sources':{'PTX_BF16_shapes':'https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma','Hopper_resources':'https://docs.nvidia.com/cuda/hopper-tuning-guide/'},'suggested_gates':['Compile canonical+fused BM128 and144 against existing BM160/176 flags and full configs; retain spills, registers, shared memory, SASS and hashes.','If resource-qualified, native H200 exactness on routing boundaries BM-1/BM/BM+1 and160/176 boundaries, changed graph routes/input, allremote and both layout paths.','Compare max-rank coherent mixed/hot distributions with paired A/A controls. Uniform all-light policy does not exercise this hypothesis.','Only explore count-conditioned cap after source/graph partition ownership and extra launch overhead are accounted for. No automatic promotion.']}
write('SM90_HEAVY_CAP_CPU_REVIEW.json',report); write('SM90_HEAVY_CAP_COMPILE_PROPOSAL.json',{'status':'PROPOSED_NO_LEASE','first_priority_bm':[128,144],'followup_if_informative':[136,152],'jobs':jobs})
aux=P.parent/'aux_speculation';snap=P/'unified_review_snapshot';snap.mkdir(exist_ok=True);pins={}
for f in ['probe_unified.py','launch_unified.py','UNIFIED_CPU_CONTRACT.json']:
 p=aux/f;data=p.read_bytes();(snap/f).write_bytes(data);pins[f]=sha(p)
write('UNIFIED_M32_INDEPENDENT_REVIEW.json',{'status':'CPU_REVIEW_NO_DECISIVE_REFERENCE_OR_CAPTURE_BLOCKER','source_sha256':pins,'positive_findings':['Exact scored scale/split/finalizer explicitly imported.','Canonical and interleaved candidates share unchanged W2 and distinct expert scales.','Four asserted substitutions preserve scored scratch alias/order and W2 semantics.','Baseline activation host observer reaches function-local import; no added GPU op.','Captured graph dispatch fixed separately for original/M2/fallback; later Python flag changes do not alter recorded kernels.','Active raw-down checks avoid invalid inactive alias assumptions; activation sentinels and all-final comparisons nonvacuous.','Changed graph input/route generations include cutoff crossing and allremote.'], 'limits':['Numerical gate only, no paired timing.','Forced architecture90 metadata rejection while running SM86 tests dispatch fallback, not native SM90 arithmetic.','Fixture H4096/I4096/E32/M32; parsed tokens/expert overrides do not extend qualification.','Launcher provenance list should additionally pin fixture bootstrap/probe_composed and Python Humming codegen/config dependencies.'],'active_source_modified':False})
print(json.dumps({'assertions':assertions,'comparisons':[{k:r[k] for k in ['baseline_bm','candidate_bm','same_weight_tile_count','higher_weight_tile_count','padding_lower_equal_higher']} for r in rows],'candidate_resources':[{k:j[k] for k in ['name','active_accumulator_values_per_thread','shared_bytes_source_estimate']} for j in jobs],'unified_review_pins':pins}))
