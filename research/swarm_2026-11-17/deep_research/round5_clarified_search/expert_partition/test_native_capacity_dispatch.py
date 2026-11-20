"""Native source-table admission follow-up; no CUDA execution."""
import ast,json,pathlib,hashlib
P=pathlib.Path(__file__).resolve().parent;R=next(p for p in P.parents if (p/'RULES.md').exists())
# Load only definitions/imports and needed setup from test, avoiding its run/write tail.
src=(P/'test_capacity_dispatch.py').read_text();tree=ast.parse(src);nodes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='results'for t in n.targets):break
 nodes.append(n)
ns={'__file__':str(P/'test_capacity_dispatch.py')};exec(compile(ast.Module(body=nodes,type_ignores=[]),'capacity_definitions','exec'),ns)
metadata=R/'swarm_2026-11-16/moe/sm90_wgmma_metadata_audit.json';tables=json.loads(metadata.read_text())['stages'];original_factory=ns['layer']
def actual_layer(cap,policy='absent',w2=None):
 x=original_factory(cap,policy,w2);x.fp32_configs={s:tables[s]['intervals']for s in ['w13','w2']};return x
ns['layer']=actual_layer;run=ns['run'];rows=[]
for cap in [32,33,40,64]:
 a=run(ns['old'],cap);off=run(ns['new'],cap);on=run(ns['new'],cap,32)
 assert a==off
 assert isinstance(on,dict)and on['cutoff']==8 and on['physical_experts']==cap and on['last_slot_count']==32 and on['invalid_preserved']
 rows.append(dict(capacity=cap,original=a,default_off=off,policy32=on))
configs={s:{label:next(c for lo,hi,c in v['intervals']if lo<m<=hi)for label,m in [('light',32),('heavy',256)]}for s,v in tables.items()}
report={'status':'PASS_NATIVE_SM90_SOURCE_TABLE_DISPATCH_ONLY','configs':configs,'cases':rows,'source_metadata':str(metadata),'source_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'scope':'Pinned architecture-mocked SM90 default intervals; M32/top8/global256. No H200 execution, no regenerated physical-capacity table, no numerical claim. Original64 happens to keep same light config here;33/40 whole-helper fallback confirmed. Physical capacity remains histogram bound.','next':'Pin or verify actual runtime fp32_configs after changing storage capacity; cannot assume native heuristics are capacity-independent.'}
(P/'NATIVE_CAPACITY_DISPATCH_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'capacities':[32,33,40,64],'cutoff':8}))
