import argparse,hashlib,json,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists())
a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args();out=P/args.out;out.mkdir(exist_ok=False)
site=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages';cuda=site/'nvidia/cu13'
cmd=[str(cuda/'bin/nvcc'),'--cubin','-arch=sm_90','-std=c++20','-DSGL_CUDA_ARCH=900','-O3','--expt-relaxed-constexpr','--ptxas-options=-v','-I'+str(P),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/include'),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/csrc/distributed'),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark'),'-I'+str(site/'tvm_ffi/include'),'-I'+str(cuda/'include/cccl'),str(P/'offline_mc_handoff.cu'),'-o',str(out/'kernel.cubin')]
(out/'command.json').write_text(json.dumps(cmd,indent=2));start=time.time();r=subprocess.run(cmd,capture_output=True,text=True,timeout=70)
(out/'compiler.stdout.log').write_text(r.stdout);(out/'compiler.stderr.log').write_text(r.stderr)
report=dict(status='OFFLINE_COMPILE_PASS_NO_RUNTIME' if r.returncode==0 else 'COMPILE_FAILED',exit_code=r.returncode,seconds=time.time()-start,sources={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['mc_workspace_handoff.cuh','offline_mc_handoff.cu','render_mc_handoff.py']},CUDA_context_created=False,GPU_runtime=False)
if r.returncode==0:report['cubin_sha256']=hashlib.sha256((out/'kernel.cubin').read_bytes()).hexdigest()
(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));sys.exit(r.returncode)
