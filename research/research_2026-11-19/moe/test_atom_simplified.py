from pathlib import Path
import subprocess,json,hashlib,sys
import compile_fp8 as cfg
here=Path(__file__).resolve().parent
args=[str(cfg.nvcc),'-std=c++17','--expt-relaxed-constexpr','-O1','-cudart=none',f'-I{here/"cutlass_pinned/include"}',*(f'-I{x}' for x in cfg.incs),str(here/'atom_simplified.cu'),'-o',str(here/'atom_simplified')]
lib=next(p for p in Path(sys.prefix).rglob('libcudart.so.*') if p.is_file())
args.extend([f'-L{lib.parent}',f'-l:{lib.name}',f'-Xlinker=-rpath={lib.parent}'])
r=subprocess.run(args,capture_output=True,text=True);assert r.returncode==0,r.stderr[:3000]
text=subprocess.check_output([str(here/'atom_simplified')],text=True);assert text.strip()=='CHECKS 2433024 BAD 0',text
out=dict(result=text.strip(),header_sha256=hashlib.sha256((here/'humming_atom.hpp').read_bytes()).hexdigest(),cpu_only=True)
(here/'atom_simplified.json').write_text(json.dumps(out,indent=2));print(out)
