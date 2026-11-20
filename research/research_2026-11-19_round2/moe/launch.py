"""Task-local launcher with installed CUDA library paths, no dependency changes."""
import os
from pathlib import Path
import sys
here=Path(__file__).resolve().parent
cuda=Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'
headers=Path(sys.prefix)/'python-headers/usr/include'
env=dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS='1',HUMMING_DISABLE_PARALLEL_BUILD='1')
for key,paths in [('PATH',[Path(sys.prefix)/'bin',cuda/'bin']),('CPATH',[headers/'python3.12',headers]),('LD_LIBRARY_PATH',[cuda/'lib','/usr/lib/wsl/lib'])]:
    env[key]=':'.join(map(str,paths))+':'+env.get(key,'')
assert sys.argv[1] in ('test_facade_gpu','test_facade_memory','test_bridge_gpu','bench_conversion','bench_hybrid_fallback','test_bridge_tiled_gpu','bench_hybrid_tiled','bench_hybrid_tiled_holdout')
os.execve(sys.executable,[sys.executable,'-u',str(here/(sys.argv[1]+'.py')),*sys.argv[2:]],env)

