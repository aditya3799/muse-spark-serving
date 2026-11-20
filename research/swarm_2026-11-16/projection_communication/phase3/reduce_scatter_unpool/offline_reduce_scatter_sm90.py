"""Compile an exact row reduce-scatter templates for SM90a; never load CUDA or cubin."""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata as metadata,json,os,shlex,subprocess,sys,time,typing
os.environ['CUDA_VISIBLE_DEVICES']=''
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--sglang-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
if os.environ.get('PHASE2_OFFLINE_BOUND')!='180':raise RuntimeError('Use bounded launcher')
args.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
jit=args.sglang_root/'python/sglang/kernels/jit';kernel=P/'reduce_scatter_rows.cuh';wrapper=P/'reduce_unpool.py'
assert sha(kernel)=='9a1b6732c360a0f228f806200002ddc6b32989ce60a1924d7808f15c9ac04bda'
assert sha(wrapper)=='54f9c07d0f4550b442bd3364303b6e48808ebdda1fb8bd3514b7f91c24540196'
spec=jit/'utils/compile/spec.py';tree=ast.parse(spec.read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FFI_INCLUDES' for t in n.targets):nodes.append(n)
 if isinstance(n,ast.FunctionDef) and n.name=='_wrapper_source':nodes.append(n)
ns={'Tuple':typing.Tuple};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(spec),'exec'),ns)
expression='MuseReduceScatterRows<bf16_t, 2, true>::run'
expression_large='MuseReduceScatterRows<bf16_t, 8, true>::run'
code=ns['_wrapper_source']((str(kernel),),(('run_2',expression),('run_4','MuseReduceScatterRows<bf16_t, 4, true>::run'),('run_8',expression_large)));unit=args.out/'kernel.cu';unit.write_text(code)
compiler_dist=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler_dist.locate_file(f)).resolve() for f in compiler_dist.files if str(f).endswith('/bin/nvcc'))
package_name=metadata.packages_distributions()['tvm_ffi'][0];ffi_dist=metadata.distribution(package_name)
ffi_header=next(Path(ffi_dist.locate_file(f)).resolve() for f in ffi_dist.files if str(f).endswith('/include/tvm/ffi/function.h'))
dlpack_header=next(Path(ffi_dist.locate_file(f)).resolve() for f in ffi_dist.files if str(f).endswith('/dlpack/dlpack.h'))
includes=[ffi_header.parents[2],dlpack_header.parents[1],jit/'include',jit/'csrc']
artifact=args.out/'kernel.sm90a.cubin';depfile=args.out/'dependencies.d'
flags=['--cubin','--gpu-architecture=sm_90a','-DSGL_CUDA_ARCH=900','-std=c++20','-O3','--expt-relaxed-constexpr','--keep','--keep-dir',str(args.out),'-Xptxas=-v','-MD','-MF',str(depfile)]
command=[str(nvcc),*flags,*['-I'+str(p) for p in includes],str(unit),'-o',str(artifact)]
version=subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,check=True).stdout
report={'status':'prepared','scope':'Offline exact row reduce-scatter compilation only; no Torch import, CUDA context/driver calls, module/cubin load, GPU execution or H200 speed claim','argv':sys.argv,'python_executable':sys.executable,'helper_sha256':sha(Path(__file__)),'launcher_sha256':sha(P/'launch_offline_reduce_scatter.py'),'architecture':'sm_90a','production_math_flags':'No extra_cuda_cflags: stock custom all-reduce uses default O3 without use_fast_math','configuration':{'world_sizes':[2,4,8],'pdl':True,'load_reduce':'original LS, graph LS and MC templates','scope':'same stock custom AR flags, uninstalled diagnostic'},'wrapper_expression':[expression,expression_large],'wrapper_generator_sha256':sha(spec),'generated_source_sha256':sha(unit),'source_hashes':{str(p):sha(p) for p in (kernel,wrapper,jit/'csrc/distributed/custom_all_reduce.cuh',jit/'include/sgl_kernel/vec.cuh')},'compiler':{'path':str(nvcc),'sha256':sha(nvcc),'version':version,'command':command,'include_dirs':[str(p) for p in includes],'ffi_distribution':package_name,'ffi_version':ffi_dist.version},'bound_seconds':180,'kill_grace_seconds':15,'torch_imported':'torch' in sys.modules}
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'compiling','expression':expression}),flush=True)
started=time.monotonic();run=subprocess.run(command,capture_output=True,text=True);report['compile_seconds']=time.monotonic()-started;report['compiler_exit_code']=run.returncode
(args.out/'stdout.log').write_text(run.stdout);(args.out/'stderr.log').write_text(run.stderr)
if run.returncode:
 report['status']='compile_failed';(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(run.returncode)
raw=artifact.read_bytes();assert raw[:4]==b'\x7fELF' and b'muse_reduce_scatter_rows_kernel' in raw,'Required reduce-scatter kernel missing from ELF'
ptx_paths=list(args.out.glob('*.ptx'));assert len(ptx_paths)==1,ptx_paths
ptx=ptx_paths[0].read_text()
required=('griddepcontrol.wait','griddepcontrol.launch_dependents')
assert all(op in ptx for op in required),'Missing PDL PTX evidence'
report['ptx_evidence']={'path':str(ptx_paths[0]),'sha256':sha(ptx_paths[0]),'required_operations':list(required),'all_present':True,'scope':'PTX operation existence only; not dataflow or runtime equivalence proof'}
deps=shlex.split(depfile.read_text().replace('\\\n',' ').split(':',1)[1]);header_hashes={}
for name in deps:
 dep=Path(name);assert dep.exists(),name;header_hashes[str(dep.resolve())]=sha(dep)
assert all(sha(Path(name))==digest for name,digest in report['source_hashes'].items()),'Source changed during compile'
assert 'torch' not in sys.modules
report.update(status='compiled',cubin_sha256=sha(artifact),cubin_bytes=artifact.stat().st_size,routing_kernel_symbol_present=True,dependency_hashes=header_hashes,dependency_digest=hashlib.sha256(json.dumps(header_hashes,sort_keys=True).encode()).hexdigest(),torch_imported=False)
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'compiled','seconds':report['compile_seconds'],'cubin_bytes':report['cubin_bytes']}),flush=True)