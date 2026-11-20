import ast,hashlib,json,pathlib,re
P=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
frozen=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/models/muse_spark.py';active=ROOT/'sglang/python/sglang/srt/models/muse_spark.py'
def methods(p):
 t=ast.parse(p.read_text(encoding='utf-8-sig'));d={}
 for cls in t.body:
  if isinstance(cls,ast.ClassDef):
   for f in cls.body:
    if isinstance(f,ast.FunctionDef):d[cls.name+'.'+f.name]=ast.dump(f,include_attributes=False)
 return d
f,a=methods(frozen),methods(active)
rows=[]
for k in f:
 if ('Attention' in k or 'DecoderLayer' in k) and k.rsplit('.',1)[1] in ['__init__','forward','bind_next_pre_attn_norm']:
  rows.append(dict(method=k,identical_AST=f[k]==a.get(k),frozen_sha256=hashlib.sha256(f[k].encode()).hexdigest(),active_sha256=hashlib.sha256(a.get(k,'').encode()).hexdigest()))
status=ROOT/'swarm_2026-11-16/coordination/status_e399c79bb1994630.json';d=json.loads(status.read_text(encoding='utf-8-sig'));s=d['result']['startup_log']['text']
lines=[l for l in s.splitlines() if re.search(r'All Reduce config|Finalize AR|CustomAllReduce|NVIDIA_H200',l)]
result=dict(status='FROZEN_SCORED_ATTENTION_SOURCE_CROSSCHECK',frozen_model_path=str(frozen.relative_to(ROOT)),frozen_model_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),active_model_sha256=hashlib.sha256(active.read_bytes()).hexdigest(),methods=rows,cached_run=dict(id=d['id'],status_sha256=hashlib.sha256(status.read_bytes()).hexdigest(),serving_flags=d['serving_flags'],communication_log_lines=lines,log_truncated=d['result']['startup_log'].get('truncated'),log_cut=d['result']['startup_log'].get('cut')),qualification='Cached scored run proves multicast/pull initialization. Per-call algorithm choices remain source-inferred, not measured telemetry. Frozen MoE/pooler paths differ and are not this candidate.')
assert all(r['identical_AST'] for r in rows if 'Attention.' in r['method']),rows
(P/'FROZEN_ATTENTION_AND_CACHED_LOG_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
