"""Render/compile one pinned Humming Hopper kernel; never load or execute it."""
from pathlib import Path
import argparse,hashlib,json,os,re,shutil,sys,time
OWN=Path(__file__).resolve().parent
P=OWN.parent
# Context-free offline compilation must not acquire the local CUDA device.
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HUMMING_DEBUG_KERNEL']='0'
os.environ['MAX_JOBS']='1'
os.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'
sys.path.insert(0,str(P/'humming_vendor'))
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['w13','w2'],required=True);p.add_argument('--tokens',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--compile',action='store_true');a=p.parse_args()
if a.tokens<=0:raise ValueError('Positive token count required')
if os.environ.get('MOE_OFFLINE_TIMEOUT_SECONDS')!='240':
    raise RuntimeError('Use launch_offline_sm90.py for the bounded TERM/KILL process group')
a.out.mkdir(parents=True,exist_ok=True)
import torch
assert not torch.cuda.is_initialized()
import humming
from humming.config import LayerConfig,ComputeConfig,TuningConfig,MmaType
from humming.kernel.humming import HummingKernel
from humming.jit.compiler import NVRTCCompiler
from humming.utils.nvrtc import get_nvrtc_library_path
assert humming.__version__=='0.1.12',humming.__version__

class OfflineHummingKernel(HummingKernel):
    def __new__(cls,*args,**kwargs):
        return object.__new__(cls)
    def __post_init__(self):
        LayerConfig.__post_init__(self)
        ComputeConfig.__post_init__(self)
        TuningConfig.__post_init__(self)
        self.sm_version=90
        self.sm_version_str='90a'
        self.cubin_loaded=False
        self.init_kernel()
    def prepare(self):
        # HummingKernel.init_kernel calls prepare after rendering. The normal
        # implementation initializes a context and loads cubin; bypass both.
        assert not torch.cuda.is_initialized()
    def load_cubin(self):
        raise AssertionError('Offline test must never load a cubin')
    @staticmethod
    def current_context():
        raise AssertionError('Offline test must never request a CUDA context')
    @staticmethod
    def _ensure_cuda_context():
        raise AssertionError('Offline test must never initialize a CUDA context')

audit_path=OWN/'sm90_grouped_metadata_audit.json'
audit=json.loads(audit_path.read_text(encoding='utf-8'))
entry=audit['stages'][a.stage]
selected=next(c for lo,hi,c in entry['intervals'] if lo<a.tokens*8<=hi)
assert selected['use_stream_k'] is False and selected['use_f16_accum'] is False
configuration=dict(entry['meta'],**selected)
configuration.update(gemm_type='grouped_masked',use_f16_accum=False,use_stream_k=False,use_cp_async=True)
kernel=OfflineHummingKernel(**configuration)
assert kernel.mma_type==MmaType.WGMMA
assert kernel.use_tma_a and kernel.use_tma_b and kernel.use_tma_c and kernel.use_warp_spec
assert kernel.is_grouped_masked_gemm
assert not kernel.use_stream_k and not kernel.use_f16_accum and not kernel.has_input_scale
assert not kernel.cubin_loaded and not torch.cuda.is_initialized()
mma=kernel.mma_op_class
assert mma.cd_dtype=='f32' and mma.val_type_cd=='float'
mma_cpp=mma.to_cpp_str()
(a.out/'mma_op_class.cuh').write_text(mma_cpp,encoding='utf-8')
(a.out/'kernel.cu').write_text(kernel.code,encoding='utf-8')
(a.out/'kernel_expression.txt').write_text(kernel.kernel_expr,encoding='utf-8')
include=P/'humming_vendor/humming/include'
headers={str(f.relative_to(include)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(include.rglob('*.cuh'))}
report={'status':'rendered','scope':'offline source generation/compile only; no CUDA context, cubin load, execution or H200 performance proof','humming_version':humming.__version__,'python_executable':sys.executable,'architecture':'sm_90a','process_timeout':{'wall_seconds':240,'kill_after_seconds':15,'executable':'/usr/bin/timeout'},'stage':a.stage,'tokens':a.tokens,'configuration':configuration,'generated_source_sha256':hashlib.sha256(kernel.code.encode()).hexdigest(),'pinned_humming_kernel_py_sha256':hashlib.sha256((P/'humming_vendor/humming/kernel/humming.py').read_bytes()).hexdigest(),'pinned_headers_digest':hashlib.sha256(json.dumps(headers,sort_keys=True).encode()).hexdigest(),'mma_type':str(kernel.mma_type),'actual_mma_op_class':{'python_class':type(mma).__name__,'shape':mma.shape,'a_dtype':mma.a_dtype,'b_dtype':mma.b_dtype,'accumulator_dtype':mma.cd_dtype,'accumulator_cpp_type':mma.val_type_cd,'cpp_sha256':hashlib.sha256(mma_cpp.encode()).hexdigest(),'ptx_operations':sorted(set(re.findall(r'(?:wgmma\.mma_async|mma\.sync)[^ \"\n;]+',mma_cpp)))},'torch_cuda_initialized':torch.cuda.is_initialized(),'resolved_features':{'tma_a':kernel.use_tma_a,'tma_b':kernel.use_tma_b,'tma_c':kernel.use_tma_c,'warp_specialization':kernel.use_warp_spec,'stream_k':kernel.use_stream_k,'f16_accum':kernel.use_f16_accum}}
(a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'status':'rendered','stage':a.stage,'tokens':a.tokens,'block_shape':kernel.block_shape,'warp_shape':kernel.warp_shape}),flush=True)
if a.compile:
    nvrtc_library=Path(get_nvrtc_library_path())
    library_hash=hashlib.sha256()
    with nvrtc_library.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            library_hash.update(chunk)
    report['compiler_provenance']={
        'nvrtc_signature':NVRTCCompiler.signature(),
        'nvrtc_library_path':str(nvrtc_library),
        'nvrtc_library_sha256':library_hash.hexdigest(),
        'flags':NVRTCCompiler.get_flags('90a',kernel.disable_fast_math),
        'include_dirs':NVRTCCompiler._get_include_dirs(),
        'external_header_limit':'CUDA/libcu++ include paths recorded; full external header closure is not hashed',
        'audit_json_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'launcher_sha256':hashlib.sha256((OWN/'launch_grouped_offline.py').read_bytes()).hexdigest(),
        'supervisor_sha256':hashlib.sha256((OWN/'launch_probe.py').read_bytes()).hexdigest(),
    }
    assert not torch.cuda.is_initialized()
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    started=time.time()
    try:
        cubin=Path(NVRTCCompiler.compile(kernel.code,sm_version='90a',kernel_expr=kernel.kernel_expr,disable_fast_math=kernel.disable_fast_math,postprocess_cubin=None))
        copied_cubin=a.out/'kernel.sm90a.cubin'
        shutil.copyfile(cubin,copied_cubin)
        report['cubin_artifact']=str(copied_cubin)
        report['cache_reused']=cubin.stat().st_mtime<started
        for logname in ('stdout.log','stderr.log','cmdline.json'):
            log=cubin.parent/logname
            if log.exists():shutil.copyfile(log,a.out/logname)
        cmdline=cubin.parent/'cmdline.json'
        if cmdline.exists():report['compiler_cmdline']=json.loads(cmdline.read_text(encoding='utf-8'))
        report.update(status='compiled',compile_seconds=time.time()-started,cubin_path=str(cubin),cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),cubin_bytes=cubin.stat().st_size)
    except Exception as exc:
        report.update(status='compile_failed',compile_seconds=time.time()-started,error=repr(exc))
        (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    assert not kernel.cubin_loaded and not torch.cuda.is_initialized()
    report['torch_cuda_initialized']=torch.cuda.is_initialized()
    (a.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'compile_seconds':report['compile_seconds'],'cubin_bytes':report['cubin_bytes']}),flush=True)
