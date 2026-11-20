from pathlib import Path
import json,hashlib
h=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/prefill')
report=dict(passed=False,scope='Stopped on first byte mismatch; preliminary stdout evidence, no full receipt because original fixture asserted before final save.',seed=2611191229,
 completed_geometry_modes=[{'real':n,'padded':p,'rope':r,'mutations':['random','integer','padded_poison'],'all_checks':True} for n,p in [(56,64),(454,512),(753,1024),(3012,4096),(3640,4096)] for r in [False,True]],
 first_failure={'real':16383,'padded':16384,'rope':False,'stride':2304,'mutation':'random','native_block':32,'candidate_block':32,'checks':{'q':True,'k':False,'v':True,'reserved':True}},
 hypotheses_unresolved=['Native padded-vs-real normalization rounding','Fused arithmetic/compiler rounding','Scatter or destination ownership'],
 source_sha256=hashlib.sha256((h/'prefill_prologue_bytes.py').read_bytes()).hexdigest(),
 decision='Prefill extension vetoed pending localization; do not promote prior passing shapes or mutate decoder frozen artifact.',
 harness_prerequisites=['Initial process failed before comparison because PATH lacked ninja/CUDA_HOME; fixed explicit LinuxPATH and installedCUDA13','Intermediate shell export failed due inheritedWindowsPATHspaces; noGPU work from that shell'])
(h/'prefill_prologue_initial_veto.json').write_text(json.dumps(report,indent=2))
