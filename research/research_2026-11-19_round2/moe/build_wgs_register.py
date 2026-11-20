import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
parent=here.parent.parent/'research_2026-11-18_target_moe/native_package_02/hopper_expert.cu'
src=parent.read_text();header=(here/'register_dequant.cuh').read_text(encoding='utf-8-sig')
needle='__device__ __forceinline__ float serving_scaled'
assert src.count(needle)==1
insertion='\n#ifndef REGISTER_DEQUANT\n#define REGISTER_DEQUANT 0\n#endif\n'+header+'\n'
port=src.replace(needle,insertion+needle)
old='const uint32_t scg = scn;'
new=old+'\n#if REGISTER_DEQUANT\n      const uint32_t coefficient_scales = register_scale_pair(scg);\n#endif'
assert port.count(old)==1;port=port.replace(old,new)
replacements=[]
for r in range(4):
 oldcall=f'af[ci & 1][jj][{r}] = serving_pair<{r}>(q, scg, table);'
 newcall=f'''#if REGISTER_DEQUANT
          af[ci & 1][jj][{r}] = register_pair<{r}, REGISTER_DEQUANT>(q, scg, coefficient_scales, table);
#else
          {oldcall}
#endif'''
 assert port.count(oldcall)==1;port=port.replace(oldcall,newcall)
 replacements.append((oldcall,newcall))
recovered=port.replace(insertion,'').replace(new,old)
for a,b in replacements:recovered=recovered.replace(b,a)
assert recovered==src
path=here/'hopper_expert_register.cu';path.write_text(port)
report=dict(parent_path=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_bytes=path.stat().st_size,macro_options={'0':'original table load','1':'exact integer normalization','2':'exact BF16 multiplication normalization'},only_changes=['header insertion','one scale conversion hoisted per scg','four coefficient calls with unchanged table branch'],recovered_original_source_exact=True,no_wgmma_routing_global_scale_activation_changes=True)
(here/'register_wgs_source_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

