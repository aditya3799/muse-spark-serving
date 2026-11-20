from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;H=P.parent/'hopper_moe';receipts=[]
for name in ['swap_wk32_micro_01','swap_wk32_pipeline_01']:
 f=H/name/'results.json';r=json.loads(f.read_text());assert r['sources_unchanged'] and r['engineered_wrong_topology_rejected'] and len(r['rows'])==13
 if 'micro'in name:assert r['swap_bf16_all_bitwise'] and r['swap_fp32_all_bitwise']
 else:assert all(v['raw']['bitwise']and v['post']['bitwise']and v['raw']['finite']and v['post']['finite']and v['inactive']for row in r['rows']for v in row['checks'].values())
 receipts.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'status':r['status'],'rows':len(r['rows'])})
f=P/'swap_full_helper_plan.py';s=f.read_text(encoding='utf-8-sig');s=s.replace("'swap_ab_humming_codegen.py'","'swap_ab_humming_codegen_v3.py'");f.write_text(s,encoding='utf-8')
f=P/'probe_swap_full_helper.py';s=f.read_text(encoding='utf-8-sig');s=s.replace('from swap_ab_humming_runtime import SwapHummingKernel','from swap_ab_humming_runtime_v3 import SwapHummingKernel').replace("'swap_ab_humming.cuh'","'swap_ab_humming_v3.cuh'").replace("'swap_ab_humming_codegen.py'","'swap_ab_humming_codegen_v3.py'").replace("'swap_ab_humming_runtime.py'","'swap_ab_humming_runtime_v3.py'");s=s.replace('# The current pinned renderer rejects requiredWK32/N4096; this block cannot run until reviewed dependency admission changes.','# Reviewed v3 admitsWK32/N4096 after independent micro/pipeline gates; this fullhelper remains a new numerical research gate and requires root lease.')
f.write_text(s,encoding='utf-8');compile(s,str(f),'exec')
(P/'SWAP_FULL_HELPER_PREREQUISITES.json').write_text(json.dumps({'status':'WK32_PREREQUISITES_VERIFIED_FULLHELPER_NOT_YET_RUN','receipts':receipts,'reviewed_backend':'Hopperv3, noactivepatch, nofourbank,N4096newfullhelperqualificationpending'},indent=2)+'\n');print(receipts)
