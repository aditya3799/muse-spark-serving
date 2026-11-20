from pathlib import Path
import hashlib
p=Path(__file__).resolve().parent/'probe_swap_full_helper.py';s=p.read_text();anchor="report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files};"
assert s.count(anchor)==1
replacement="extra_sources=[ROOT/'optimization_marlin_tiles_2026-11-15/kernel_runtime.py',OLD/'phase2/fixture_tuning_cache.py',Path(actmod.__file__),ROOT/'sglang/python/sglang/kernels/jit/csrc/elementwise/activation.cuh']\n files+=extra_sources\n depdir=a.out/'dependency_snapshots';depdir.mkdir(exist_ok=True)\n for i,f in enumerate(extra_sources):(depdir/('%02d_'%i+f.name)).write_bytes(f.read_bytes())\n report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files};"
s=s.replace(anchor,replacement);p.write_text(s);compile(s,str(p),'exec');print(hashlib.sha256(p.read_bytes()).hexdigest())
