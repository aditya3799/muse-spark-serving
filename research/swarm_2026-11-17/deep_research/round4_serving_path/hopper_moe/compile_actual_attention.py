import hashlib,json,pathlib,subprocess,sys,time
P=pathlib.Path(__file__).resolve().parent;ROOT=next(p for p in P.parents if (p/'RULES.md').exists());out=P/'offline_actual_01';out.mkdir(exist_ok=False)
site=pathlib.Path(sys.prefix)/'lib/python3.12/site-packages';cuda=site/'nvidia/cu13';rows=[]
for name,source in [('push_actual','virtual_attention_actual.cu'),('mc_actual','offline_mc_actual.cu')]:
 job=out/name;job.mkdir();cmd=[str(cuda/'bin/nvcc'),'--cubin','-arch=sm_90a','-std=c++20','-DSGL_CUDA_ARCH=900','-O3','--expt-relaxed-constexpr','--ptxas-options=-v','-I'+str(P),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/include'),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/csrc/distributed'),'-I'+str(ROOT/'sglang/python/sglang/kernels/jit/csrc/muse_spark'),'-I'+str(site/'tvm_ffi/include'),'-I'+str(cuda/'include/cccl'),str(P/source),'-o',str(job/'kernel.cubin')]
 (job/'command.json').write_text(json.dumps(cmd,indent=2));start=time.time();r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(job/'compiler.stdout.log').write_text(r.stdout);(job/'compiler.stderr.log').write_text(r.stderr)
 row=dict(name=name,source=source,source_sha256=hashlib.sha256((P/source).read_bytes()).hexdigest(),exit_code=r.returncode,seconds=time.time()-start)
 if r.returncode==0:row['cubin_sha256']=hashlib.sha256((job/'kernel.cubin').read_bytes()).hexdigest()
 rows.append(row);(out/'results.json').write_text(json.dumps(dict(status='OFFLINE_ACTUAL_BRANCH_ONLY',CUDA_context=False,jobs=rows),indent=2));assert r.returncode==0,r.stderr[-2000:]
print(json.dumps(rows,indent=2))
