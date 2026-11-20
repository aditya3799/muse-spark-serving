"""CPU JSON/AST audit only: deduplicate existing and needed SM90 configurations."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

OWN=Path(__file__).resolve().parent
SWARM=OWN.parent
MOE=SWARM/'moe'
PROJ=SWARM/'projection_communication'
COORD=SWARM/'coordination'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit_path=MOE/'sm90_wgmma_metadata_audit.json'
audit=json.loads(audit_path.read_text(encoding='utf-8'))
include=MOE/'humming_vendor/humming/include'
header_hashes={p.relative_to(include).as_posix():sha(p) for p in sorted(include.rglob('*.cuh'))}
headers_digest=hashlib.sha256(json.dumps(header_hashes,sort_keys=True).encode()).hexdigest()
for rel,digest in audit['source_hashes'].items():
    assert sha(MOE/'humming_vendor/humming'/rel)==digest,(rel,'metadata source changed')
ns={}
exec(compile((PROJ/'ep_tuning_selector.py').read_text(encoding='utf-8-sig'),'<CPU selector>','exec'),ns)
tree=ast.parse((PROJ/'adaptive_expert_split.py').read_text(encoding='utf-8-sig'))
planner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_partition_configs')
exec(compile(ast.Module(body=[planner],type_ignores=[]),'<CPU planner>','exec'),ns)
layer=NS(w13_post_scale=NS(numel=lambda:32),fp32_configs={s:v['intervals'] for s,v in audit['stages'].items()})
combined_path=PROJ/'adaptive_split_v3/serving_export/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
combined_root=PROJ/'adaptive_split_v3'
combined_manifest=json.loads((combined_root/'manifest.json').read_text(encoding='utf-8'))
for rel,digest in combined_manifest['sources'].items():
    assert sha(combined_root/'serving_export'/rel)==digest
nvrel='python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
oldtree=ast.parse((MOE/'candidate_v3/serving_export'/nvrel).read_text(encoding='utf-8'))
newtree=ast.parse((combined_root/'serving_export'/nvrel).read_text(encoding='utf-8'))
oldhelper=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='_humming_expert_outputs')
newhelper=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name=='_humming_expert_outputs_default')
newhelper.name=oldhelper.name
assert ast.dump(oldhelper)==ast.dump(newhelper),'Combined default helper changed'
combined_ns={'deepcopy':deepcopy}
combined_tree=ast.parse(combined_path.read_text(encoding='utf-8-sig'))
combined_nodes=[n for n in combined_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('select_ep_config','choose_partition_configs')]
exec(compile(ast.Module(body=combined_nodes,type_ignores=[]),'<exact combined CPU planner>','exec'),combined_ns)
for m in range(1,513):
    assert ns['choose_partition_configs'](layer,m*8,256)==combined_ns['choose_partition_configs'](layer,m*8,256)
ns=combined_ns

def configuration(stage,tuning):
    result=dict(audit['stages'][stage]['meta'],**tuning)
    result.update(gemm_type='indexed',use_f16_accum=False,use_stream_k=False,use_cp_async=True)
    return result
def signature(config):
    return hashlib.sha256(json.dumps({'architecture':'sm_90a','configuration':config},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def select(stage,m):
    return next(c for lo,hi,c in layer.fp32_configs[stage] if lo<m*8<=hi)

existing=[]
covered={}
for stage in ('w13','w2'):
    for m in (1,128,16384):
        folder=(COORD/('offline_sm90_'+stage+'_prefill')) if m==16384 else MOE/('offline_sm90_'+stage+'_m'+str(m))
        report_path=folder/'report.json'; receipt_path=folder/'run_receipt.json'
        r=json.loads(report_path.read_text()); receipt=json.loads(receipt_path.read_text())
        cubin=folder/'kernel.sm90a.cubin'
        assert r['status']=='compiled' and receipt['exit_code']==0
        assert receipt['report_sha256']==sha(report_path)
        assert r['stage']==stage and r['tokens']==m and not r['torch_cuda_initialized']
        assert sha(cubin)==r['cubin_sha256'] and cubin.stat().st_size==r['cubin_bytes']
        assert r['configuration']==configuration(stage,select(stage,m))
        assert r['pinned_headers_digest']==headers_digest
        assert r['pinned_humming_kernel_py_sha256']==sha(MOE/'humming_vendor/humming/kernel/humming.py')
        assert r['compiler_provenance']['audit_json_sha256']==sha(audit_path)
        assert sha(MOE/'offline_sm90_compile.py')==r['compiler_provenance']['helper_sha256']
        assert sha(MOE/'launch_offline_sm90.py')==r['compiler_provenance']['launcher_sha256']
        sig=signature(r['configuration'])
        entry={'stage':stage,'representative_tokens':m,'signature':sig,'directory':str(folder),
               'report_sha256':sha(report_path),'receipt_sha256':sha(receipt_path),
               'cubin_sha256':sha(cubin),'cubin_bytes':cubin.stat().st_size,'exit_code':0,
               'configuration':r['configuration']}
        existing.append(entry);covered.setdefault(sig,[]).append(str(folder))

requests={}
typical=[]
for m in sorted(set(range(1,513))|{16384}):
    plan=ns['choose_partition_configs'](layer,m*8,256) if m<=512 else None
    arms={'default':{stage:select(stage,m) for stage in ('w13','w2')}} if plan is None else {arm:plan[arm] for arm in ('light','heavy')}
    for arm,stages in arms.items():
        for stage,tuning in stages.items():
            config=configuration(stage,tuning);sig=signature(config)
            entry=requests.setdefault(sig,{'signature':sig,'stage':stage,'configuration':config,'users':[],
                                          'covered_by':covered.get(sig,[])})
            entry['users'].append({'tokens':m,'arm':arm})
            if m in (1,8,16,32,33,128,256,512,16384):
                typical.append({'tokens':m,'arm':arm,'stage':stage,'signature':sig,'block_shape':config['block_shape'],
                                'covered':sig in covered})

missing=[]
for sig,r in requests.items():
    if r['covered_by']:continue
    # The unchanged compiler selects the default interval for tokens*8. Find
    # an exact equivalent default representative, even for adaptive light arms.
    candidates=[m for m in range(1,513) if signature(configuration(r['stage'],select(r['stage'],m)))==sig]
    assert candidates,'No representative accepted by unchanged compiler'
    preferred=next((m for m in (16,32,256) if m in candidates),candidates[0])
    out=OWN/('offline_sm90_'+r['stage']+'_m'+str(preferred))
    linux_out='/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/attention_speculation/'+out.name
    priority=1 if preferred in (16,32,256) else 2
    missing.append(dict(r,representative_tokens=preferred,priority=priority,output_directory=str(out),
                        argv=['/home/aditya/.venvs/okc-marlin-tiles/bin/python',
                              '/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe/launch_offline_sm90.py',
                              '--stage',r['stage'],'--tokens',str(preferred),'--out',linux_out,'--compile']))
missing.sort(key=lambda r:(r['priority'],r['representative_tokens'],r['stage']))
manifest={'status':'CPU_PREPARED_WAITING_FOR_EXPLICIT_COMPILER_HANDOFF',
          'scope':'Native Python metadata/source/artifact audit only; no Torch/Humming import, CUDA context, WSL process, compile or execution',
          'geometry':{'local_experts':32,'global_experts':256,'hidden':4096,'intermediate':4096,'topk':8},
          'coverage_domain':'Every adaptive token count M1..512, plus default M16384; no claim for other layer geometry or all larger default intervals',
          'source_hashes':{str(p):sha(p) for p in (audit_path,PROJ/'adaptive_expert_split.py',PROJ/'ep_tuning_selector.py',combined_path,MOE/'offline_sm90_compile.py',MOE/'launch_offline_sm90.py')},
          'combined_planner_parity_cases':512,
          'combined_default_helper_ast_preserved':True,
          'signature_definition':'SHA256 of canonical sorted compact JSON {architecture:sm_90a,configuration:<complete layer+compute+tuning config>}; source/compiler hashes checked separately',
          'existing_verified_artifacts':existing,'existing_unique_signatures':len(covered),
          'requested_unique_signatures':len(requests),'missing_unique_signatures':len(missing),
          'priority1_missing_count':sum(r['priority']==1 for r in missing),'representative_coverage':typical,
          'missing':missing,'execution_rules':['Do not launch before explicit handoff after GPU/JIT tests finish.',
          'Use existing bounded launcher/helper unchanged; one serial compile at a time.',
          'Archive each clean process exit and hash report/cubin; do not count rendered/timed-out outputs as compiled.',
          'No cubin load, CUDA context, kernel execution, runtime correctness or H200 speed claim.']}
with (OWN/'SM90_ADAPTIVE_COMPILE_COVERAGE.json').open('w',encoding='utf-8',newline='\n') as handle:
    handle.write(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k in ('status','existing_unique_signatures','requested_unique_signatures','missing_unique_signatures','priority1_missing_count')},indent=2))
print(json.dumps([{'stage':r['stage'],'tokens':r['representative_tokens'],'priority':r['priority'],'block_shape':r['configuration']['block_shape'],'signature':r['signature']} for r in missing if r['priority']==1],indent=2))
