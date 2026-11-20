"""Actual upstream ring startup predicates plus logical read-set counterexample.

No tensor kernel, model checkpoint, GPU memory saving or serving speed is tested.
"""
import ast,hashlib,json,sys,types
from pathlib import Path
from typing import Any,Callable,Optional
here=Path(__file__).resolve().parent;source=here/'ring_sources/dflash_draft_ring.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
names={'dflash_draft_reads_rows_by_position','draft_kv_ring_unsupported_reason','_mha_ring_unsupported_reason','draft_kv_ring_rows_per_req','draft_kv_ring_paged_rows_per_req','draft_kv_ring_pool_rows'}
env=dict(Any=Any,Callable=Callable,Optional=Optional,DFLASH_MHA_RING_DRAFT_BACKENDS=('trtllm_mha','fa4'),DFLASH_MLA_ONLY_DRAFT_BACKENDS=('cutedsl_mla',),dflash_dcp_mla_draft_backend=lambda:'cutedsl_mla')
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names],type_ignores=[]),'pinned-ring-predicates','exec'),env)
utils=types.ModuleType('sglang.srt.speculative.dflash_utils')
utils.get_dflash_layer_types=lambda cfg:cfg['layer_types']
utils.parse_dflash_draft_config=lambda *,draft_hf_config:types.SimpleNamespace(is_mla=draft_hf_config['mla'])
sys.modules[utils.__name__]=utils
config=types.ModuleType('sglang.srt.configs.model_config');config.mla_attention_dcp_size=lambda cfg,**kw:kw['dcp_size'];sys.modules[config.__name__]=config
cfg=dict(mla=False,layer_types=['sliding_attention']*5)
base=dict(load_draft_hf_config=lambda:cfg,draft_backend='fa3')
cases=[('native-fa3',{},True),('fa4-mha',dict(draft_backend='fa4'),False),
 ('fa4-explicit-window',dict(draft_backend='fa4',draft_window_size=2048),True),
 ('fa4-unified',dict(draft_backend='fa4',enable_unified_memory=True),True),
 ('fa4-dp',dict(draft_backend='fa4',enable_dp_attention=True),True),
 ('fa4-dcp',dict(draft_backend='fa4',dcp_size=2),True),
 ('fa4-pd-nixl',dict(draft_backend='fa4',disaggregation_mode='decode',disaggregation_transfer_backend='nixl'),True)]
results=[]
for name,kw,decline in cases:
 reason=env['draft_kv_ring_unsupported_reason'](**dict(base,**kw));assert (reason is not None)==decline
 results.append(dict(case=name,rejected=decline,reason=reason))
# For equal scores, attention's weighted output is the average of its values.
# Cached prefix0..11, new suffix12..15, final window8..15: a ring whose floor
# equals prefix hit12 omits valid old keys8..11. This establishes a possible
# numerical difference; it is not an empirical statement about this checkpoint.
native_positions=list(range(8,16));ring_positions=list(range(12,16))
counterexample=dict(prefix_hit=12,new_suffix_length=4,window_size=8,
 native_positions=native_positions,ring_positions=ring_positions,
 equal_score_native_output=sum(native_positions)/len(native_positions),
 equal_score_ring_output=sum(ring_positions)/len(ring_positions),
 conclusion='Different read sets can change draft attention/acceptance. Target speculative verification may preserve final greedy tokens, but this is not unchanged drafting.')
assert counterexample['equal_score_native_output']!=counterexample['equal_score_ring_output']
report=dict(scope=__doc__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),predicate_cases=results,counterexample=counterexample,
 ring_rows_example=env['draft_kv_ring_paged_rows_per_req'](window_left=2047,block_size=8,page_size=1),
 actual_model_window_confirmed=False,eligible_native_fa3=False)
(here/'ring_applicability_cpu.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
