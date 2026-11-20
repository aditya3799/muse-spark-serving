from pathlib import Path
import ast
here=Path(__file__).resolve().parent
s=(here/'compile_wgs_register.py').read_text(encoding='utf-8-sig')
s=s.replace("source=here/'hopper_expert_register.cu'","source=here/'hopper_expert_register_finite.cu'")
s=s.replace('for variant in (1,2):','for variant in (2,):').replace("f'-DREGISTER_DEQUANT={variant}',","f'-DREGISTER_DEQUANT={variant}','-DREGISTER_FINITE=1',")
s=s.replace('wgs_register_{variant}','wgs_finite_{variant}').replace('wgs_register_compile.json','wgs_finite_compile.json').replace("len(report['variants'])==2","len(report['variants'])==1")
s=s.replace('baseline_compile_succeeded_prior_to_disassembler_path_failure=True, ','finite_scan_required=True, ')
ast.parse(s);(here/'compile_wgs_finite.py').write_text(s,encoding='utf-8')
