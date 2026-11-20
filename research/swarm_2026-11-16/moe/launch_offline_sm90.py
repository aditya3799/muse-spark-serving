from pathlib import Path
import importlib.metadata as metadata,os,sys
compiler=metadata.distribution('nvidia-cuda-nvcc')
nvcc=next(Path(compiler.locate_file(f)).resolve() for f in compiler.files if str(f).endswith('/bin/nvcc'))
cuda=nvcc.parent.parent
os.environ['CUDA_HOME']=str(cuda)
os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(cuda/'bin')+os.pathsep+os.environ.get('PATH','')
os.environ['LD_LIBRARY_PATH']=str(cuda/'lib64')+':/usr/lib/wsl/lib:'+os.environ.get('LD_LIBRARY_PATH','')
os.environ['SGLANG_CACHE_DIR']=str(Path.home()/'.cache/okc-marlin-tiles')
os.environ['SGLANG_JIT_CACHE_DIR']=str(Path.home()/'.cache/okc-marlin-tiles/jit')
os.environ['MAX_JOBS']='1'
os.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'
os.environ['PYTHONUNBUFFERED']='1'
headers=Path(sys.prefix)/'python-headers/usr/include'
os.environ['CPATH']=str(headers/'python3.12')+':'+str(headers)+':'+os.environ.get('CPATH','')
script=Path(__file__).parent/'offline_sm90_compile.py'
import json
print(json.dumps({'python_executable':sys.executable,'nvcc_executable':str(nvcc),'max_jobs':os.environ['MAX_JOBS'],'serial_nvrtc':os.environ['HUMMING_DISABLE_PARALLEL_BUILD']}),flush=True)
os.environ['MOE_OFFLINE_TIMEOUT_SECONDS']='240'
os.execv('/usr/bin/timeout',['/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s',sys.executable,str(script)]+sys.argv[1:])
