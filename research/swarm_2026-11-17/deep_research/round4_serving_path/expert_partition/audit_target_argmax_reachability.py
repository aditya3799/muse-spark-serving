import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
H=Path(__file__).resolve().parent;R=next(p for p in H.parents if (p/'sglang/python/sglang/srt').is_dir());S=R/'sglang/python/sglang/srt'
jobp=R/'swarm_2026-11-16/coordination/status_e399c79bb1994630.json';job=json.loads(jobp.read_text(encoding='utf-8-sig'));assert job['result']['flags']==[]
def load(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 class NoImports(ast.NodeTransformer):
  def visit_Import(self,n): return None
  def visit_ImportFrom(self,n): return None
 nodes=[NoImports().visit(n) for n in nodes];module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);ast.fix_missing_locations(module);exec(compile(module,str(path),'exec'),ns)
class Backend:
 def is_none(self):return True
p=NS(attn_dp_enabled=False,disable_attn_tp_gather=False,moe_dense_tp_size=None,tp_size=8,num_dp_ranks=1)
ns={'get_parallel':lambda:p,'MoeA2ABackend':Backend,'get_moe_a2a_backend':Backend,'get_exec':lambda:NS(moe=NS(elastic_ep_backend=None))}
load(S/'utils/common.py',['require_mlp_tp_gather','require_attn_tp_gather'],ns)
mlp=ns['require_mlp_tp_gather']();attn=ns['require_attn_tp_gather']();assert (mlp,attn)==(False,False)
ns2={'is_hip':lambda:False,'ForwardMode':NS(TARGET_VERIFY='verify',DECODE='decode')}
load(S/'layers/attention/graph_variants.py',['create_attention_graph_variants','create_dsv41_candidate_graph_variants'],ns2)
mr=NS(model_config=NS(hf_text_config=NS(model_type='muse_spark')),spec_algorithm=NS(is_dspark=lambda:False),is_draft_worker=False)
assert ns2['create_attention_graph_variants'](object()) is None
assert ns2['create_dsv41_candidate_graph_variants'](mr,'verify',8) is None
ns3={};load(S/'model_executor/runner/decode_cuda_graph_runner.py',['_lora_backend_skips_inactive','ragged_verify_compact_graphs_enabled'],ns3)
assert not ns3['_lora_backend_skips_inactive'](NS(lora_manager=None))
assert not ns3['ragged_verify_compact_graphs_enabled'](NS(supports_ragged_verify=lambda:False))
fields={};wanted={'moe_a2a_backend','moe_dense_tp_size','pp_size','dp_size','attn_dp_size','attn_cp_size','dcp_size','enable_dp_attention','enable_pdmux','enable_two_batch_overlap','enable_lora','disable_overlap_schedule','enable_waterfill'}
for path in (S/'arg_groups/fields').glob('*.py'):
 for n in ast.walk(ast.parse(path.read_text())):
  if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in wanted:fields[n.target.id]={'value':ast.literal_eval(n.value),'path':str(path.relative_to(S)),'line':n.lineno}
report={'status':'SOURCE_REACHABLE_FOR_DECLARED_JUDGE_CONFIG_NOT_LIVE_PROOF','job':'e399c79bb1994630','job_flags':job['result']['flags'],'explicit_candidate_override':{'--moe-runner-backend':'humming'},'override_sources':['swarm_2026-11-16/coordination/round5_humming_adaptive_fused/scored_submission_plan.json','swarm_2026-11-16/coordination/round5_humming_adaptive_fused/scored_submission_receipt.json'],'job_sha256':hashlib.sha256(jobp.read_bytes()).hexdigest(),'judge_source':'RULES.md lines72-80 base TP8/EP8, Marlin, DFLASH width8; scored candidate explicitly overrides moe-runner-backend=humming per plan/receipt. Aggregate result flags[] is incomplete metadata, not evidence of no override.','defaults':fields,'source_extracted_checks':{'require_mlp_tp_gather':mlp,'require_attn_tp_gather':attn,'require_mlp_sync':bool(p.attn_dp_enabled or mlp or attn),'attention_graph_variants':None,'dsv41_candidate_graph_variants':None,'record_nolora_graph_default':False,'ragged_verify_mode':False},'topology':'Runtime derives attn_dcp_size1 when dcp_size1; attn_dp_size1 and no elastic join yields attn_dp_enabledFalse; TPGroup exposes ranks/rank_in_group/world_size. EP8 alone does not imply attention DP or require_gathered_buffer.','backend':'resolve_decode_backend uses FullCudaGraphBackend for default FULL, cuda non-out-of-tree; record_nolora False without manager; DFLASH sets TARGET_VERIFY,width8.','stream':'model_runner.forward_stream created at line469; tp_worker.get_worker_info exports it to scheduler; overlap scheduler encloses whole DFlash call in forward_stream_ctx at4671. Target forward and consumption occur sequentially inside same context; capture private stream is deliberately not compared.','limits':['Saved server_args line is truncated before topology fields; cannot assert a live guard census from that line.','Source reachability combines official local judge flags, explicit Humming candidate override and pinned defaults; no model construction or native runner executed.','Runtime count of logits processors, actual LP scalars, actual graph backend and capture hooks still need positive logged admission in a serving probe.','Changes in configuration, topology, stream or LP state reject consumption; no widening to unsupported cases.'],'decision':'Not structurally disabled for the declared scored Muse TP8 configuration. Still require actual admitted/rejected replay counters before any performance claim.'}
(H/'TARGET_ARGMAX_GUARD_REACHABILITY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['source_extracted_checks'],indent=2))
