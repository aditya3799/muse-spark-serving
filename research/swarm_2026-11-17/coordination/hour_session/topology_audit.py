import ast, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REPO=ROOT/'sglang'
BASE='297e54fd105e974cdf401e5b4d20253a22c16f90'
OUT=Path(__file__).resolve().parent
FILES={
 'defaults':'python/sglang/srt/configs/muse_spark.py',
 'widths':'python/sglang/srt/runtime_context.py',
 'speculation':'python/sglang/srt/arg_groups/speculative_hook.py',
 'parallel':'python/sglang/srt/arg_groups/parallel_hook.py',
 'dp_attention':'python/sglang/srt/layers/dp_attention.py',
 'embedding':'python/sglang/srt/layers/vocab_parallel_embedding.py',
 'attention_linear':'python/sglang/srt/layers/muse_spark/muse_spark_layers.py',
 'kv_budget':'python/sglang/srt/mem_cache/kv_cache_configurator.py',
}
def source(path):
 return subprocess.check_output(['git','-C',str(REPO),'show',BASE+':'+path]).decode('utf-8')
sources={name:source(path) for name,path in FILES.items()}
model_path=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/models/muse_spark.py'
model=model_path.read_text(encoding='utf-8')
cls=next(n for n in ast.parse(sources['defaults']).body if isinstance(n,ast.ClassDef) and n.name=='MuseSparkTextConfig')
init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
defaults={a.arg:ast.literal_eval(d) for a,d in zip(init.args.args[-len(init.args.defaults):],init.args.defaults)}
H=defaults['hidden_size']; Q=defaults['num_attention_heads']*defaults['head_dim']; V=defaults['num_key_value_heads']*defaults['head_dim']; L=defaults['num_hidden_layers']; vocab=defaults['vocab_size']; E=defaults['moe_hidden_size']; I=defaults['moe_intermediate_size']; experts=defaults['num_local_experts']
# Q and output gate are query-sized. K and V share the KV width. All attention projections BF16.
attention_global_bytes=2*(H*(2*Q+2*V)+Q*H)
embedding_and_head_global_bytes=2*2*vocab*H
local_expert_layer_bytes=(experts//8)*3*E*I*(.5+1/16)
def mib(n):return n/(1<<20)
def gib(n):return n/(1<<30)
rows=[]
for dp in (1,2,4,8):
 atp=8//dp
 rows.append({'attention_dp':dp,'attention_tp':atp,'moe_ep':8,'moe_tp':1,
  'attention_extra_weights_MiB_per_layer':mib(attention_global_bytes*(1/atp-1/8)),
  'embedding_and_lm_head_extra_weights_GiB':gib(embedding_and_head_global_bytes*(1/atp-1/8)),
  'illustrative_default62layers_extra_weights_GiB':gib((L*attention_global_bytes+embedding_and_head_global_bytes)*(1/atp-1/8)),
  'scope':'Target BF16 attention/embedding/head only; excludes draft, vision, buffers, loading transients, allocator. Source defaults, not authenticated checkpoint config.',
  'kv_budget_effect':'Extra resident weights reduce automatically sized KV pool; weight delta alone does NOT establish peak-memory-rule failure.',
  'one_request_attention_active_GPUs':atp,
  'balanced_attention_flops_relative_to_baseline_per_GPU':1.0,
  'one_request_attention_flops_relative_to_baseline_active_GPU':float(dp),
  'expert_work':'Same global gathered token batch on all EP ranks; no expert FLOP reduction from attention DP.',
  'source_eligibility':'DFLASH requires enable_dp_lm_head, CP1 and no DP speculative/prefill coordination; these are necessary, not startup/runtime validation.' if dp>1 else 'Scored layout',
  'tradeoffs':['Additional DP hidden gather/scatter','Higher per-GPU projection weight traffic','Scored row-projection helpers require attentionDP1','Possible balanced scheduling or collective efficiency gains require measurement'] if dp>1 else []})
report={'status':'SOURCE_AUDIT_AND_RESOURCE_MODEL_NOT_RUNTIME_VALIDATED','baseline_commit':BASE,
 'source_hashes':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()},
 'scored_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
 'defaults_scope':'Pinned configuration constructor defaults only. Actual model layer count and full/sliding layer schedule have not been authenticated.',
 'constructor_defaults':{k:defaults[k] for k in ('hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','vocab_size','moe_hidden_size','moe_intermediate_size','num_local_experts')},
 'attention_dp_options':rows,
 'other_topologies':[
  {'option':'TP8 EP4 MoE-DP1','decision':'Rejected by model constructor: NVFP4 experts require MoE-TP1; derived MoE-TP2 is unsupported.'},
  {'option':'TP8 EP4 MoE-DP2','decision':'Not a free expert sharding change. Local expert count doubles; generic resolver also couples MoE-DP with attention CP. DFlash topology/correctness not verified.', 'extra_expert_weights_GiB_per_sparse_layer':gib(local_expert_layer_bytes), 'illustrative62sparse_layer_extra_GiB':gib(local_expert_layer_bytes*L)},
  {'option':'Pipeline parallel >1','decision':'Rejected by DFLASH argument handler; requires a substantive worker/model pipeline implementation.'},
  {'option':'Attention DP plus CP>1','decision':'Rejected by DFLASH handler.'},
  {'option':'Standalone prefill/context parallel','decision':'Not certified. Generic CUDA hook alone is not model/backend integration proof; requires KV/attention, token layout and MoE review.'}
 ],
 'scored_path_effect':{'DP_MoE':'_forward_dp_attention gathers global hidden, runs global experts, fullTP AR, replicated post projection, then scatters local rows. It bypasses _reduce_and_unpool and its row-projection optimizations.', 'source_evidence':['MuseSpark1xMoE.__init__','MuseSpark1xMoE._forward_dp_attention','MuseSpark1xMoE._can_shard_prefill_row_projection','_handle_dflash','KVCacheConfigurator._profile_available_bytes']},
 'decision':'Do not blindly spend a scarce check on DP2/DP4. It is a supported structural avenue with larger model weights and unchanged expert work, and may hurt low-load latency. Need a balanced/imbalanced critical-path model and full memory/TTFT/correctness checks before promotion.',
 'measured_serving_gain':None}
(OUT/'TOPOLOGY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'attention_DP2_extra_weight_GiB_illustrative_L62':rows[1]['illustrative_default62layers_extra_weights_GiB'],'DP2_expert_work_reduction':0,'EP4_MoEDP2_extra_GiB_illustrative_L62':gib(local_expert_layer_bytes*L),'runtime_tested':False}))