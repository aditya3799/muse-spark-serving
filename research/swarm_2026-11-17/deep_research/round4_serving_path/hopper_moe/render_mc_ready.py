import hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent
src=(P/'mc_workspace_handoff.cuh').read_text()
a=src.index('struct ResearchMCWorkspaceHandoff');b=src.index('} // namespace sglang',a)
body=src[a:b].replace('ResearchMCWorkspaceHandoff','ResearchMCReadyPlaneHandoff',1)
body=body.replace('CHECK_HOST(input_begin+nbytes<=plane_begin || plane_begin+nbytes<=input_begin);','CHECK_HOST(input_begin==plane_begin);')
body=body.replace('  cuda_memcpy(local_workspace,in.data_ptr());','  // The caller already wrote all projection elements into the local plane on this stream.')
body=body.replace('// Input remains a separate complete projection. Direct-to-plane GEMM is NOT enabled.','// This separate entry accepts ONLY a complete same-stream GEMM result already in the local plane.')
code='#pragma once\n#include "mc_workspace_handoff.cuh"\nnamespace sglang {\n'+body+'}\n'
(P/'mc_ready_plane_handoff.cuh').write_bytes(code.encode())
cu=(P/'offline_mc_handoff.cu').read_text().replace('#include "mc_workspace_handoff.cuh"','#include "mc_ready_plane_handoff.cuh"',1)
(P/'offline_mc_ready.cu').write_bytes(cu.encode())
meta=dict(status='HELD_DIRECT_PRODUCER_SOURCE_ONLY',sha256=hashlib.sha256(code.encode()).hexdigest(),base_adapter_sha256=hashlib.sha256(src.encode()).hexdigest(),changes=['Input/output must both equal current local plane pointer','No input staging copy; native MC body unchanged'],admission_extra=['TORCH/AUTO BF16 projection backend only; biasNone; noGEMV/custom/backendoverride','sameM,N,K,strides,alignment as qualified baseline; noTF32/precision settings changes','full output initialization by current GEMM before collective, on same stream','H200 producer equivalence must pass on all ranks before collective comparison'],local_evidence='projection_plane_01:20 strictpairs actualK1024/N8192; SM86 only',native_status='uncompiled/unexecuted at generation')
(P/'MC_READY_PLANE_CONTRACT.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
