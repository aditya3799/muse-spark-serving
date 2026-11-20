import ast,difflib,hashlib,json
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
relative='python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
base=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources'/relative
baseline=base.read_bytes();original=baseline.decode()
source=P/'scalable_partition.py';candidate=source.read_text(encoding='utf-8-sig')
line='from sglang.kernels.ops.moe.muse_spark_humming_split import choose_partition_configs as original_choose_partition_configs, partition_routes'
assert line in candidate
candidate=candidate.replace(line,'original_choose_partition_configs = choose_partition_configs')
candidate=candidate.replace('def choose_partition_configs(', 'def _choose_large_prefill_configs(')
candidate=candidate.replace('plan = choose_partition_configs(layer,valid,global_experts)', 'plan = _choose_large_prefill_configs(layer,valid,global_experts)')
candidate=candidate.replace('def split_expert_outputs_scalable(', 'def _split_large_prefill_expert_outputs(')
wrapper='''

def split_expert_outputs(original_helper, layer, x, local_ids, torch, global_experts=None,
                         *, retain=False, defer_downscale=False):
    # Research export: disabled until actual target-device qualification.
    # This opt-in layer attribute is not enabled by any server/source change.
    if (getattr(layer, "_humming_large_prefill_partition_enabled", False) is not True
            or local_ids.numel() <= 4096):
        return _original_split_expert_outputs(
            original_helper, layer, x, local_ids, torch, global_experts,
            retain=retain, defer_downscale=defer_downscale)
    return _split_large_prefill_expert_outputs(
        original_helper, layer, x, local_ids, torch, global_experts,
        retain=retain, defer_downscale=defer_downscale)
'''
new=original+'\n\n# Isolated research alternative; the scored implementation above is preserved.\n_original_split_expert_outputs = split_expert_outputs\n\n'+candidate+wrapper
encoded=new.encode();assert encoded[:len(baseline)]==baseline
out=P/'candidate_export';path=out/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encoded)
base_tree=ast.parse(original);new_tree=ast.parse(new)
old_fn=next(n for n in base_tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_expert_outputs')
new_fn=[n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_expert_outputs'][-1]
assert ast.dump(old_fn.args)==ast.dump(new_fn.args)
assert max(map(len,new.splitlines()))<=400
# Dispatch guards only: actual device validation is recorded separately.
rows=[]
def marker(label):
 def call(*args,**kwargs):return label,args,kwargs
 return call
ns={'_original_split_expert_outputs':marker('original'),'_split_large_prefill_expert_outputs':marker('large')}
exec(compile(ast.Module(body=[new_fn],type_ignores=[]),str(path),'exec'),ns)
for state in ('absent',False,True,1,'true',None):
 for routes in (0,1,4096,4097,131072):
  layer=SimpleNamespace()
  if state!='absent':setattr(layer,'_humming_large_prefill_partition_enabled',state)
  ids=SimpleNamespace(numel=lambda routes=routes:routes)
  result=ns['split_expert_outputs']('helper',layer,'x',ids,'torch',256,retain=True,defer_downscale=True)
  expected='large' if state is True and routes>4096 else 'original'
  assert result[0]==expected
  assert result[1]==('helper',layer,'x',ids,'torch',256)
  assert result[2]=={'retain':True,'defer_downscale':True}
  rows.append({'enable_state':state,'routes':routes,'dispatch':result[0],'passed':True})
diff=''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
(out/'incremental.diff').write_bytes(diff.encode())
manifest={'status':'DEFAULT_OFF_RESEARCH_EXPORT_NOT_SUBMISSION_READY','baseline_file':str(base),'baseline_sha256':hashlib.sha256(baseline).hexdigest(),'prototype_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'export_file':relative,'export_sha256':hashlib.sha256(encoded).hexdigest(),'incremental_diff_sha256':hashlib.sha256((out/'incremental.diff').read_bytes()).hexdigest(),'bytes':len(encoded),'diff_bytes':len(diff.encode()),'baseline_byte_prefix_preserved':True,'public_signature_unchanged':True,'default_enabled':False,'opt_in_attribute':'layer._humming_large_prefill_partition_enabled is True','small_route_dispatch_original':True,'max_line_length':max(map(len,new.splitlines())),'guard_checks':len(rows),'guard_rows':rows,'evidence':['../K_POLICY_CPU_AUDIT.json','../k_preserve_m1024_01/results.json','../original_policy_negative_01/results.json','../k_preserve_boundaries_01/results.json','../fullrow_recurrence_01/results.json','../RANK_COHERENT_REVIEW.json','../partition_memcheck_01/run_receipt.json','../partition_racecheck_01/run_receipt.json'],'pending':['Target H200 configuration legality and raw/final numerical gates','SM90 changed-input graph execution and memory/race validation','Model-level outputs, peakmemory, TTFT and serving frontier','Explicit integration decision after qualification; no enable call is included'],'root_patch_unchanged':True,'active_checkout_modified':False,'submissions':0}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k not in ('guard_rows','evidence','pending')},indent=2))

