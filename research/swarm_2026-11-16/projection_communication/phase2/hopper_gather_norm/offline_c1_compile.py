"""Compile an exact production C1 template for SM90a; never load CUDA or cubin."""
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata as metadata,json,os,shlex,subprocess,sys,time,typing
os.environ['CUDA_VISIBLE_DEVICES']=''
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--sglang-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--threads',type=int,choices=(256,512),required=True);args=parser.parse_args()
if os.environ.get('PHASE2_OFFLINE_BOUND')!='180':raise RuntimeError('Use bounded launcher')
args.out.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
jit=args.sglang_root/'python/sglang/kernels/jit';kernel=jit/'csrc/muse_spark/all_gather_norm_residual.cuh';norm=jit/'csrc/muse_spark/norm_residual.cuh'
assert sha(kernel)=='cb53a903365ac8688864c7835253ea8723d777d7ebb49b2c12ba140a6f956186'
assert sha(norm)=='c227348c4d705183d377ac623a62b1ed6957eb8832fb0508f4ac99f1aa3cb0c8'
spec=jit/'utils/compile/spec.py';tree=ast.parse(spec.read_text());nodes=[]
for n in tree.body:
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_FFI_INCLUDES' for t in n.targets):nodes.append(n)
 if isinstance(n,ast.FunctionDef) and n.name=='_wrapper_source':nodes.append(n)
ns={'Tuple':typing.Tuple};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(spec),'exec'),ns)
prefetch=args.threads==256
expression='MuseSparkAllGatherNormResidual<8192, 8, '+str(args.threads)+', 1, true, true, '+str(prefetch).lower()+', true, true>::run'
code=ns['_wrapper_source']((str(kernel),),(('run',expression),));unit=args.out/'kernel.cu';unit.write_text(code)
compiler_dist=metadata.distribution('nvidia-cuda-nvcc');nvcc=next(Path(compiler_dist.locate_file(f)).resolve() for f in compiler_dist.files if str(f).endswith('/bin/nvcc'))
package_name=metadata.packages_distributions()['tvm_ffi'][0];ffi_dist=metadata.distribution(package_name)
ffi_header=next(Path(ffi_dist.locate_file(f)).resolve() for f in ffi_dist.files if str(f).endswith('/include/tvm/ffi/function.h'))
dlpack_header=next(Path(ffi_dist.locate_file(f)).resolve() for f in ffi_dist.files if str(f).endswith('/dlpack/dlpack.h'))
includes=[ffi_header.parents[2],dlpack_header.parents[1],jit/'include']
artifact=args.out/'kernel.sm90a.cubin';depfile=args.out/'dependencies.d'
flags=['--cubin','--gpu-architecture=sm_90a','-DSGL_CUDA_ARCH=900','-std=c++20','-O3','--expt-relaxed-constexpr','-Xptxas=-v','-MD','-MF',str(depfile)]
command=[str(nvcc),*flags,*['-I'+str(p) for p in includes],str(unit),'-o',str(artifact)]
version=subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,check=True).stdout
report={'status':'prepared','scope':'Offline exact production C1 compilation only; no Torch import, CUDA context/driver calls, module/cubin load, GPU execution or H200 speed claim','argv':sys.argv,'python_executable':sys.executable,'helper_sha256':sha(Path(__file__)),'launcher_sha256':sha(P/'launch_offline_c1.py'),'architecture':'sm_90a','configuration':{'hidden':8192,'world':8,'threads':args.threads,'cluster_size':1,'pdl':True,'post_norm_is_input_scale':True,'prefetch_weights':prefetch,'has_post_weight':True,'has_gate':True},'wrapper_expression':expression,'wrapper_generator_sha256':sha(spec),'generated_source_sha256':sha(unit),'source_hashes':{str(p):sha(p) for p in (kernel,norm)},'compiler':{'path':str(nvcc),'sha256':sha(nvcc),'version':version,'command':command,'include_dirs':[str(p) for p in includes],'ffi_distribution':package_name,'ffi_version':ffi_dist.version},'bound_seconds':180,'kill_grace_seconds':15,'torch_imported':'torch' in sys.modules}
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'compiling','threads':args.threads,'expression':expression}),flush=True)
started=time.monotonic();run=subprocess.run(command,capture_output=True,text=True);report['compile_seconds']=time.monotonic()-started;report['compiler_exit_code']=run.returncode
(args.out/'stdout.log').write_text(run.stdout);(args.out/'stderr.log').write_text(run.stderr)
if run.returncode:
 report['status']='compile_failed';(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');raise SystemExit(run.returncode)
raw=artifact.read_bytes();assert raw[:4]==b'\x7fELF' and b'muse_spark_all_gather_norm_residual_row_kernel' in raw,'Required C1 kernel missing from ELF'
deps=shlex.split(depfile.read_text().replace('\\\n',' ').split(':',1)[1]);header_hashes={}
for name in deps:
 dep=Path(name);assert dep.exists(),name;header_hashes[str(dep.resolve())]=sha(dep)
assert 'torch' not in sys.modules
report.update(status='compiled',cubin_sha256=sha(artifact),cubin_bytes=artifact.stat().st_size,c1_kernel_symbol_present=True,dependency_hashes=header_hashes,dependency_digest=hashlib.sha256(json.dumps(header_hashes,sort_keys=True).encode()).hexdigest(),torch_imported=False)
(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'compiled','threads':args.threads,'seconds':report['compile_seconds'],'cubin_bytes':report['cubin_bytes']}),flush=True)