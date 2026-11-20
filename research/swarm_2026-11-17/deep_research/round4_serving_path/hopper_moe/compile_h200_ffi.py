import pathlib,json,hashlib,subprocess,time,sys
P=pathlib.Path(__file__).resolve().parent
out=P/'offline_ffi_01';out.mkdir(exist_ok=False)
source='\n'.join(['#include <tvm/ffi/container/tensor.h>','#include <tvm/ffi/dtype.h>','#include <tvm/ffi/error.h>','#include <tvm/ffi/extra/c_env_api.h>','#include <tvm/ffi/function.h>','#include "h200_mc_gate.cuh"','namespace sglang {']+['TVM_FFI_DLL_EXPORT_TYPED_FUNC(%s, (%s));'%(a,b) for a,b in [('local_address','research_local_pull_address'),('staged','ResearchMCWorkspaceHandoff::run'),('direct','ResearchMCReadyPlaneHandoff::run'),('control','AllReduceKernel<bf16_t,8,false>::run'),('norm_small','MuseSparkNormResidual<8192,256,false,true,1,false,false>::run'),('norm_large','MuseSparkNormResidual<8192,512,false,false,1,false,false>::run')]]+['}'])+'\n'
(out/'ffi_wrapper.cu').write_text(source)
cmd=json.loads((P/'offline_actual_01/mc_actual/command.json').read_text());cmd[1]='-c';cmd[-3]=str(out/'ffi_wrapper.cu');cmd[-1]=str(out/'ffi_wrapper.o');cmd.extend(['-Xcompiler','-fPIC'])
(out/'command.json').write_text(json.dumps(cmd,indent=2));t=time.time();r=subprocess.run(cmd,capture_output=True,text=True,timeout=80)
(out/'compiler.stdout.log').write_text(r.stdout);(out/'compiler.stderr.log').write_text(r.stderr)
report=dict(status='OFFLINE_FULL_FFI_OBJECT_COMPILE_ONLY',CUDA_context=False,exit_code=r.returncode,seconds=time.time()-t,sources={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['h200_mc_gate.cuh','h200_mc_gate.py','mc_ready_plane_handoff.cuh','mc_workspace_handoff.cuh']},wrapper_sha256=hashlib.sha256(source.encode()).hexdigest(),limitations=['No link/load/FFI call','No H200 execution','No graph capture or native MC performance'])
if r.returncode==0:report['object_sha256']=hashlib.sha256((out/'ffi_wrapper.o').read_bytes()).hexdigest()
(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));print(r.stderr[-1800:]);sys.exit(r.returncode)
