import json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
LANES=ROOT/'swarm_2026-11-16'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
 reports=[]
 for lane in ['moe','projection_communication','attention_speculation']:
  folder=LANES/lane/'phase4_hour'
  for p in sorted(folder.glob('*/run_receipt.json')):
   j=read(p);r=p.parent/'results.json'
   row={'lane':lane,'directory':str(p.parent.relative_to(ROOT)),'receipt_sha256':sha(p),'exit_code':j.get('exit_code'),'receipt_status':j.get('status')}
   if r.exists():
    result=read(r);row.update(report_sha256=sha(r),report_status=result.get('status'))
    expected=j.get('report_sha256')
    if expected:row['receipt_report_hash_matches']=expected==row['report_sha256']
   reports.append(row)
 baseline=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate.diff'
 assert sha(baseline)=='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b'
 assert sha(ROOT/'patch.diff')=='cf02617f790e28c9229ac9aec5e517f04344c49491347ef732a352fbaa3aa4a4'
 mismatches=[r for r in reports if r.get('receipt_report_hash_matches') is False]
 j={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RECOVERY_EVIDENCE_INDEX','baseline_patch_sha256':sha(baseline),'root_patch_sha256':sha(ROOT/'patch.diff'),'scope':'Indexes actual exit receipts, including failures. A successful offline compiler or diagnostic is not H200 serving validation. Original failed runs remain present.','runs':reports,'report_hash_mismatches':mismatches,'handoffs':{}}
 for lane,name in [('moe','HANDOFF.json'),('projection_communication','HOUR_HANDOFF.json'),('attention_speculation','HOUR_RESEARCH_REPORT.json')]:
  p=LANES/lane/'phase4_hour'/name
  if p.exists():j['handoffs'][lane]={'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
 (OUT/'EVIDENCE_INDEX.json').write_text(json.dumps(j,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'receipts':len(reports),'mismatches':mismatches,'baseline_hash_unchanged':True,'root_patch_hash_unchanged':True}))
if __name__=='__main__':build()
