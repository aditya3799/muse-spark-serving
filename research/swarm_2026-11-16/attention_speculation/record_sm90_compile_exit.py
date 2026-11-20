"""Archive an observed compiler exit and validate its offline artifact provenance."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

P=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['w13','w2'],required=True)
p.add_argument('--tokens',choices=[16,32,256],type=int,required=True)
p.add_argument('--session-id',type=int,required=True)
p.add_argument('--exit-code',type=int,required=True)
a=p.parse_args()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((P/'SM90_ADAPTIVE_COMPILE_COVERAGE.json').read_text(encoding='utf-8'))
target=next(r for r in manifest['missing'] if r['priority']==1 and r['stage']==a.stage and r['representative_tokens']==a.tokens)
out=Path(target['output_directory']);out.mkdir(exist_ok=True)
receipt={'observed_at_utc':datetime.now(timezone.utc).isoformat(),'session_id':a.session_id,'exit_code':a.exit_code,
         'stage':a.stage,'tokens':a.tokens,'signature':target['signature'],'argv':target['argv'],
         'scope':'Offline SM90a source compilation only; no context/cubin load/kernel execution/H200 runtime claim.'}
report_path=out/'report.json'
if report_path.exists():
    receipt['report_sha256']=sha(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    receipt['compiler_report_status']=report['status']
    if a.exit_code==0:
        assert report['status']=='compiled' and report['architecture']=='sm_90a'
        assert report['stage']==a.stage and report['tokens']==a.tokens
        assert report['configuration']==target['configuration']
        assert report['torch_cuda_initialized'] is False
        assert report['actual_mma_op_class']['accumulator_dtype']=='f32'
        assert report['configuration']['use_f16_accum'] is False and report['configuration']['use_stream_k'] is False
        old=json.loads((Path(manifest['existing_verified_artifacts'][0]['directory'])/'report.json').read_text())
        for key in ('pinned_humming_kernel_py_sha256','pinned_headers_digest'):
            assert report[key]==old[key]
        for key in ('helper_sha256','launcher_sha256','audit_json_sha256','nvrtc_library_sha256'):
            assert report['compiler_provenance'][key]==old['compiler_provenance'][key]
        cubin=out/'kernel.sm90a.cubin'
        assert sha(cubin)==report['cubin_sha256'] and cubin.stat().st_size==report['cubin_bytes']
        receipt.update(artifact_verified=True,cubin_sha256=sha(cubin),cubin_bytes=cubin.stat().st_size,
                       compile_seconds=report.get('compile_seconds'),cache_reused=report.get('cache_reused'),
                       cuda_context_initialized=False)
else:
    assert a.exit_code!=0,'Successful exit without report is not accepted'
with (out/'run_receipt.json').open('w',encoding='utf-8',newline='\n') as h:
    h.write(json.dumps(receipt,indent=2)+'\n')
receipts=[]
for item in manifest['missing']:
    if item['priority']!=1:continue
    rp=Path(item['output_directory'])/'run_receipt.json'
    if rp.exists():receipts.append(json.loads(rp.read_text(encoding='utf-8')))
passed=sum(r.get('artifact_verified',False) and r['exit_code']==0 for r in receipts)
summary={'status':'COMPLETE_OFFLINE_PRIORITY_COVERAGE' if passed==6 else 'IN_PROGRESS_OR_FAILED',
         'completed_verified_configs':passed,'priority_config_count':6,'runs':receipts,
         'coverage_manifest_sha256':sha(P/'SM90_ADAPTIVE_COMPILE_COVERAGE.json'),
         'scope':'Six selected configuration compiles only; additional generic config and all H200 runtime/serving gaps remain.'}
with (P/'SM90_PRIORITY_COMPILE_RESULTS.json').open('w',encoding='utf-8',newline='\n') as h:
    h.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(receipt,indent=2));print('Verified priority configurations: %s/6'%passed)
