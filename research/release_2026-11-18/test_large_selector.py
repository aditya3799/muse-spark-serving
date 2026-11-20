import ast, copy, hashlib, json, types
from pathlib import Path
P=Path(__file__).resolve().parent
BASE=P.parent/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
module=ast.parse(BASE.read_text(encoding='utf-8'))
chosen=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')]
ns={'deepcopy':copy.deepcopy};exec(compile(ast.Module(body=chosen,type_ignores=[]),str(BASE),'exec'),ns)
original=ns['choose_partition_configs']
def signature(c):
    return tuple(sorted((k,tuple(v) if isinstance(v,(list,tuple)) else v) for k,v in c.items()))
ns.update(original_choose_partition_configs=original,_large_prefill_config_signature=signature)
exec(compile((P/'large_partition_selector.py.txt').read_text(),str(P/'large_partition_selector.py.txt'),'exec'),ns)
select=ns['_choose_large_prefill_configs']
class Scale:
    def numel(self):return 32
def fixture(light_m=32,heavy_m=128):
    light={'block_shape':[light_m,256,128],'warp_shape':[16,64,64],'use_f16_accum':False,'use_stream_k':False,'num_stages':3}
    heavy={'block_shape':[heavy_m,256,64],'warp_shape':[32,64,64],'use_f16_accum':False,'use_stream_k':False,'num_stages':3}
    layer=types.SimpleNamespace(w13_post_scale=Scale(),w2_post_scale=Scale(),fp32_configs={s:[(0,1024,copy.deepcopy(light)),(1024,131072,copy.deepcopy(heavy))] for s in ('w13','w2')})
    admit(layer);return layer
def admit(layer):
    entries=[]
    for s,table in layer.fp32_configs.items():
        for _,_,heavy in table:
            entries.append((s,signature(heavy)))
            for _,_,light in table:
                repaired=copy.deepcopy(light)
                repaired['block_shape'][2]=heavy['block_shape'][2];repaired['warp_shape'][2]=heavy['warp_shape'][2]
                entries.append((s,signature(repaired)))
    layer._humming_large_prefill_configs=frozenset(entries)
checks=[]
def check(name,condition):
    assert condition,name
    checks.append(name)
l=fixture();before=copy.deepcopy(l.fp32_configs);r=select(l,6144,256)
check('valid_plan',r is not None)
check('K_repair',all(r['light'][s]['block_shape'][2]==r['heavy'][s]['block_shape'][2] and r['light'][s]['warp_shape'][2]==r['heavy'][s]['warp_shape'][2] for s in ('w13','w2')))
check('input_not_mutated',l.fp32_configs==before)
r['light']['w13']['block_shape'][0]=999
check('result_not_alias',l.fp32_configs==before)
for routes in (1,32,4096):
    l=fixture();del l._humming_large_prefill_configs
    check('small_original_'+str(routes),select(l,routes,256)==original(l,routes,256))
mutations={
 'missing_setup':lambda l:delattr(l,'_humming_large_prefill_configs'),
 'mutable_admission':lambda l:setattr(l,'_humming_large_prefill_configs',set(l._humming_large_prefill_configs)),
 'empty_admission':lambda l:setattr(l,'_humming_large_prefill_configs',frozenset()),
 'missing_w2':lambda l:l.fp32_configs.pop('w2'),
 'extra_stage':lambda l:l.fp32_configs.update(extra=l.fp32_configs['w2']),
 'missing_interval':lambda l:l.fp32_configs['w2'].pop(),
 'overlap':lambda l:l.fp32_configs['w2'].append((1000,8000,copy.deepcopy(l.fp32_configs['w2'][0][2]))),
 'partial_config':lambda l:l.fp32_configs['w13'][0][2].pop('warp_shape'),
 'mutated_signature':lambda l:l.fp32_configs['w13'][0][2].update(num_stages=4),
 'zero_shape':lambda l:l.fp32_configs['w13'][0][2]['block_shape'].__setitem__(0,0),
 'bool_shape':lambda l:l.fp32_configs['w13'][0][2]['block_shape'].__setitem__(0,True),
 'nondivisible':lambda l:l.fp32_configs['w13'][0][2]['warp_shape'].__setitem__(0,7),
 'f16':lambda l:l.fp32_configs['w13'][0][2].update(use_f16_accum=True),
 'streamk':lambda l:l.fp32_configs['w13'][1][2].update(use_stream_k=True),
 'malformed_interval':lambda l:l.fp32_configs['w13'].append(('bad',9000,{})),
 'partial_admission':lambda l:setattr(l,'_humming_large_prefill_configs',frozenset(v for v in l._humming_large_prefill_configs if v[0]=='w13')),
}
for name,mutate in mutations.items():
    l=fixture();mutate(l);check(name,select(l,6144,256) is None)
for routes in (131073,True,'6144',None):check('bad_route_'+str(routes),select(fixture(),routes,256) is None)
for global_e in (0,31,255,True,None):check('bad_global_'+str(global_e),select(fixture(),6144,global_e) is None)
# Post-repair full equality requires a new check despite pre-repair distinctness.
l=fixture(128,128)
for s,table in l.fp32_configs.items():table[0][2]['warp_shape'][0]=32
admit(l);check('postrepair_noop',select(l,6144,256) is None)
# Still reject static oversubscription even if an admission stub mistakenly lists it.
l=fixture()
for table in l.fp32_configs.values():
    for _,_,c in table:c['warp_shape']=[1,8,32]
admit(l);check('threads_over1024',select(l,6144,256) is None)
l=fixture(256,128);admit(l);check('row_increase',select(l,6144,256) is None)
report={'status':'PASS_CPU_SOURCE_EXTRACTED_SELECTOR','checks':len(checks),'names':checks,'selector_sha256':hashlib.sha256((P/'large_partition_selector.py.txt').read_bytes()).hexdigest(),'original_source_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'scope':'Actual scored chooser AST plus new selector; signature stub only. Native admission separately owned by Hopper; no GPU/compiler/serving claim.'}
(P/'evidence/LARGE_SELECTOR_CPU_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
