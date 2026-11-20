import pathlib,hashlib,json,datetime
P=pathlib.Path(__file__).resolve().parent;D=P/'pr50800_frozen_v2';assert not D.exists();D.mkdir()
base=P/'pr50800_frozen/manifest.json';basehash=hashlib.sha256(base.read_bytes()).hexdigest();assert basehash=='c99de5a627e8ff1f1a5ae16148d9c31254006bf8d227b08b4418f8394d3984dc'
src=P.parent/'coordination/wgmma_hazard_audit/pr50800_boundaries/complete_cfg_audit.json';raw=src.read_bytes();newhash=hashlib.sha256(raw).hexdigest();assert newhash=='ca91762ff5270aca3a48ae4be7149fe30c829b438a446f54d43d6f2f78979f05'
out=D/'boundary_corrected_cfg_audit.json';out.write_bytes(raw)
old=P/'pr50800_frozen/boundary/corrected_cfg_audit.json'
r=dict(frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',kind='Superseding audit overlay; source/code/proofs unchanged.',base_manifest=dict(path='../pr50800_frozen/manifest.json',sha256=basehash),replacement=dict(base_relative_path='boundary/corrected_cfg_audit.json',superseded_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),path=out.name,sha256=newhash,source=str(src)),requires_both_base_and_overlay=True,reason='Base freeze copied the earlier vendor-CFG audit before final reachability/edge hardening. Earlier receipt remains immutable as evidence.',source_changes=False,kernel_changes=False,rerun_compile_or_gpu=False,submission_package=False,conclusive_2x_evidence=False,submission_authorized=False,readiness='HOLD; corrected static known-hazard screen only. H200 math/progress/memory/wholeFFN/serving gates remain missing.')
(D/'manifest.json').write_text(json.dumps(r,indent=2));overlay=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest()
f=P/'ROUND2_MOE_EVIDENCE.json';index=json.loads(f.read_text());index['updated_utc']=datetime.datetime.utcnow().isoformat()+'Z';index['grouped_study_effective_freeze']=dict(base_manifest='pr50800_frozen/manifest.json',base_sha256=basehash,required_overlay='pr50800_frozen_v2/manifest.json',overlay_sha256=overlay)
index['evidence'].append(dict(path='pr50800_frozen_v2/manifest.json',sha256=overlay,scope='Mandatory hardened-boundary-CFG replacement overlay',status='Supersedes only earlier copied audit; base source/proofs immutable; staticonly'))
for e in index['evidence']:
 if e['path']=='pr50800_frozen/manifest.json':e['status']='FROZEN source/proof bundle; MUST apply v2 audit overlay; NOTsubmissionpackage'
f.write_text(json.dumps(index,indent=2));print(json.dumps(dict(overlay_manifest_sha256=overlay,index_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),new_receipt_sha256=newhash),indent=2))
