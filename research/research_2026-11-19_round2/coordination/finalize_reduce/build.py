import pathlib,sys,os,subprocess,resource,time,json,hashlib
HERE=pathlib.Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
os.environ['CUDA_VISIBLE_DEVICES']=''
cuda=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13';arch=sys.argv[1] if len(sys.argv)>1 else '86'
args=[str(cuda/'bin/nvcc'),'-std=c++17','-O3','-arch=sm_'+arch,'-Xptxas=-v','--shared','-Xcompiler=-fPIC','-I'+str(cuda/'include/cccl'),str(HERE/'combine_reduce.cu'),'-o',str(HERE/('combine_sm'+arch+'.so')),'-L'+str(cuda/'lib'),'-l:libcudart.so.13','-Xlinker=-rpath','-Xlinker='+str(cuda/'lib')]
start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=120)
(HERE/('build_sm'+arch+'.log')).write_text(r.stdout+r.stderr)
receipt={'returncode':r.returncode,'command':args,'seconds':time.monotonic()-start,'rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,'source_sha256':hashlib.sha256((HERE/'combine_reduce.cu').read_bytes()).hexdigest()}
(HERE/('build_sm'+arch+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));print((r.stdout+r.stderr)[-1800:]);raise SystemExit(r.returncode)
