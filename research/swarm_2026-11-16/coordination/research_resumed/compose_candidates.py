from pathlib import Path
import ast, hashlib, json, shutil, subprocess
S=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=S/'coordination/round5_humming_adaptive_fused/candidate_sources'
ROUTER=S/'projection_communication/phase2/router_reconstruct_fusion/scored_router_candidate/candidate_sources'
NORM=S/'coordination/phase2/round1_norm_glue/candidate_sources'
SWA=S/'attention_speculation/phase2/swa_scheduler_reuse/glue_swa_snapshot/candidate_sources'
MODEL='python/sglang/srt/models/muse_spark.py'
ATTN='python/sglang/srt/layers/attention/flashattention_backend.py'
COMM='python/sglang/srt/layers/communication/muse_spark_gemm_ag.py'
HELPER='python/sglang/srt/layers/attention/fa3_swa_scheduler.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pins={BASE/MODEL:'255b2ce03ac0ecc500e230c74b8f3966144bf7e8009b99bd77d77dd9a470f0c6',ROUTER/MODEL:'a0ee703c9871a79cddca41c1405d0612d657a1e7249e06605383ee6cf6a5152c',NORM/MODEL:'c3d1a9ec68d5cbf0749780e3c76a01f203a1b9e2d9cf22e9024ddc16e3fd9d33',NORM/ATTN:'0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e',NORM/COMM:'309c9e34fdf3a8aebe8ab2bea9126316cd154804dd5e3c831fde18bd8d395986',SWA/ATTN:'8ec4afbf430d925dac7da2ea30463e924727b23c316110aeac89db9e9e8e85db',SWA/HELPER:'48bfd175863d9b2669f2db2e03fb48e2b63e34b2d8d048199fd9664455b7a113'}
for p,h in pins.items(): assert sha(p)==h,(p,'source changed')
router_validation=json.loads((OUT/'router_candidate/package_validation.json').read_text())
for rel,h in router_validation['source_hashes'].items():assert sha(ROUTER/rel)==h
merge=subprocess.run(['git','merge-file','-p',str(ROUTER/MODEL),str(BASE/MODEL),str(NORM/MODEL)],capture_output=True)
assert merge.returncode==0,merge.stderr.decode()
merged=merge.stdout
assert b'<<<<<<<' not in merged
ast.parse(merged)
def methods(data):
    tree=ast.parse(data)
    return {(c.name,f.name):ast.dump(f,include_attributes=False) for c in tree.body if isinstance(c,ast.ClassDef) for f in c.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
bm,rm,nm,mm=[methods(data) for data in ((BASE/MODEL).read_bytes(),(ROUTER/MODEL).read_bytes(),(NORM/MODEL).read_bytes(),merged)]
assert bm.keys()==rm.keys()==nm.keys()==mm.keys()
changed_norm=[k for k in bm if bm[k]!=nm[k]]
changed_router=[k for k in bm if bm[k]!=rm[k]]
assert {k[1] for k in changed_norm}=={'_forward_experts'}
assert {k[1] for k in changed_router}=={'_forward_experts','forward_split2'}
assert {k[1] for k in bm if bm[k]!=mm[k]}=={'_forward_experts','forward_split2'}
for k in bm:
    if k[1]=='forward_split2':assert mm[k]==rm[k]
    elif k[1]!='_forward_experts':assert mm[k]==bm[k]
for variant,include_swa in [('norm_glue_router',False),('norm_glue_router_swa',True)]:
    dest=OUT/variant/'candidate_sources'
    for rel in router_validation['source_hashes']:
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROUTER/rel).read_bytes())
    (dest/MODEL).write_bytes(merged)
    (dest/COMM).write_bytes((NORM/COMM).read_bytes())
    (dest/ATTN).write_bytes(((SWA if include_swa else NORM)/ATTN).read_bytes())
    if include_swa:(dest/HELPER).write_bytes((SWA/HELPER).read_bytes())
    sources={}
    for p in sorted(dest.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            if p.suffix=='.py':ast.parse(p.read_bytes())
            sources[p.relative_to(dest).as_posix()]=sha(p)
    record={'status':'COMPOSED_SOURCE_REVIEW_REQUIRED_H200_PENDING','variant':variant,'include_swa':include_swa,'base_scored_job':'e399c79bb1994630','base_score':1.2237,'required_flag':'--moe-runner-backend=humming','source_pins':{str(p):h for p,h in pins.items()},'sources':sources,'shared_model_merge':'git merge-file clean; only original norm and router modifications; all other model methods AST identical','merge_exit':merge.returncode,'merged_model_sha256':hashlib.sha256(merged).hexdigest(),'excluded':['Scale+activation fusion with near-parity fullchain evidence','Rejected grouped MoE','Rejected row-sharded router','Cached persistent metadata with mixed wrapper timing'],'limits':['Individual components locally validated; composed full model untested','SWA optional variant needs H200 FP8/native graph validation','No remote submission or serving score assigned']}
    (dest.parent/'composition.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'variant':variant,'files':len(sources),'merged_model_sha256':record['merged_model_sha256']}))