from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent
r=json.loads((p/'frozen_recovered_m1/results.json').read_text(encoding='utf-8'))
assert r['passed'] and len(r['rows'])==4
assert r['artifact_sha256']=='bc19ab1afd56f204b85c54ff4d80f22bb6c8214e75d978eff37cdef1eeeb0277'
replays=[v for row in r['rows'] for v in row['replays']]
assert len(replays)==12 and all(x['passed'] for x in replays)
a=json.loads((p/'HARNESS_AUDIT.json').read_text(encoding='utf-8'))
for n,h in a['harness_hashes'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
a['status']='native audit passed; exact frozen E4 H4096 M1 bounded CUDA proof passed and exited0'
a['gpu_proof']={'result':'frozen_recovered_m1/results.json','session':20569,'exit_code':0,'cases':4,'replays':12,'max_relative_l2':max([row['eager_relative_l2'] for row in r['rows']]+[x['relative_l2'] for x in replays]),'max_absolute_error':max([row['eager_max_abs'] for row in r['rows']]+[x['max_abs'] for x in replays]),'allremote_exact_zero':True,'limits':'E4 hot_rank uses uniform fallback; E32 true hot-rank, M8+, memory, timing, and H200 validation remain pending'}
(p/'HARNESS_AUDIT.json').write_bytes((json.dumps(a,indent=2)+'\n').encode('utf-8'))
f=p/'RUN_NEXT.txt';s=f.read_text(encoding='utf-8-sig');s+='\nRecovery update: Ubuntu restart explicitly authorized and completed byroot. Exact frozen M1/E4/H4096 proof now PASSED/exited0, fourcases/12replays; see HARNESS_AUDIT.json. GPUreleased toprojection, thenattention; MoE M8+ waitsforrootgrant. No pending delayedprocess-controlcommand orMoEcompiler. Memoryprobe still unexecuted. The preflight now enforces both frozen hashes and independentstockFP32 source before GPUimports; default launcher metadata logs exactPython/nvcc paths.\n';f.write_bytes(s.replace('\r\n','\n').encode('utf-8'))
print(json.dumps(a['gpu_proof'],indent=2))
