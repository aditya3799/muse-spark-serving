import pathlib,sys,os,subprocess,resource,time,json,hashlib,importlib.util
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
arch=sys.argv[1] if len(sys.argv)>1 else '86'
resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
try:pathlib.Path('/proc/self/coredump_filter').write_text('0')
except OSError:pass
os.environ['CUDA_VISIBLE_DEVICES']=''
site=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages';cuda=site/'nvidia/cu13';tvm=pathlib.Path(importlib.util.find_spec('tvm_ffi').origin).parent
includes=[ROOT/'sglang/python/sglang/kernels/jit/include',tvm/'include',tvm/'3rdparty/dlpack/include',cuda/'include/cccl']
args=[str(cuda/'bin/nvcc'),'-std=c++20','-O3','-DSGL_CUDA_ARCH='+str(int(arch.rstrip('a'))*10),'-gencode=arch=compute_'+arch+',code=sm_'+arch,'--expt-relaxed-constexpr','-Xptxas=-v','-shared','-Xcompiler=-fPIC',*[('-I'+str(x)) for x in includes],str(HERE/'sim_native.cu'),'-o',str(HERE/('sim_sm'+arch+'.so'))]
start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,timeout=180)
(HERE/('build_sm'+arch+'.log')).write_text(r.stdout+r.stderr)
receipt=dict(command=args,returncode=r.returncode,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,memory_cap=2*1024**3,source_sha256=hashlib.sha256((HERE/'sim_native.cu').read_bytes()).hexdigest())
(HERE/('build_sm'+arch+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print((r.stdout+r.stderr)[-2500:]);raise SystemExit(r.returncode)
