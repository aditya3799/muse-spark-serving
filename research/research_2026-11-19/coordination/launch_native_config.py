import os,pathlib,sys
here=pathlib.Path(__file__).resolve().parent
cuda=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'
headers=pathlib.Path(sys.prefix)/'python-headers/usr/include'
env=dict(os.environ,CUDA_HOME=str(cuda),CUDA_PATH=str(cuda),MAX_JOBS='1',HUMMING_DISABLE_PARALLEL_BUILD='1')
for key,paths in (('PATH',[pathlib.Path(sys.prefix)/'bin',cuda/'bin']),('CPATH',[headers/'python3.12',headers]),('LD_LIBRARY_PATH',[cuda/'lib','/usr/lib/wsl/lib'])):
 env[key]=':'.join(map(str,paths))+':'+env.get(key,'')
pathlib.Path('/proc/self/coredump_filter').write_text('0')
os.execve(sys.executable,[sys.executable,'-u',str(here/'native_config_fixture.py'),*sys.argv[1:]],env)
