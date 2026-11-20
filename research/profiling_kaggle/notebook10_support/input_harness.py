"""Input/check helpers extracted unchanged from validated Notebook 05."""
import torch
if 'ExperimentCore' not in globals():
    from timing_support import ExperimentCore,routes,reference,expert_mapping
if 'BaselineSourceSlice' not in globals():
    from source_slices import BaselineSourceSlice,CandidateSourceSlice

class SourceInputs(ExperimentCore):
    def pair(self,no_map=False):
        kwargs={'num_experts':256,'num_local_experts':256 if no_map else 32,'local_expert_offset':0 if no_map else 64}
        device='cuda:0' if self.gpu else 'cpu'
        return BaselineSourceSlice(**kwargs).to(device),CandidateSourceSlice(**kwargs).to(device)

    def inputs(self,n,dtype='int64',strided=False):
        device='cuda:0' if self.gpu else 'cpu'
        raw=torch.empty((n,16 if strided else 8),dtype=getattr(torch,dtype),device=device)
        raw=raw[:,::2] if strided else raw
        return (torch.ones((n,32),device=device,dtype=torch.float16),torch.full((n,8),0.125,device=device,dtype=torch.float16),raw)

    def set_inputs(self,pair,inputs,n,pattern='uniform',kind='contiguous',seed=604):
        values=routes(n,pattern,seed);mapping=expert_mapping(kind,seed)
        inputs[2].copy_(torch.tensor(values,dtype=inputs[2].dtype).reshape(n,8))
        if pair[0]._expert_map is not None:
            new=torch.tensor(mapping,dtype=torch.int32,device=inputs[2].device)
            pair[0]._expert_map.copy_(new);pair[1].update_expert_map_(new)
        return values,mapping if pair[0]._expert_map is not None else None

    def check_result(self,output,module,inputs,values,mapping):
        fused,local,weights=output
        ids=inputs[2]
        expected_fused=(mapping is not None and inputs[0].is_cuda and ids.is_contiguous() and module._expert_map.is_contiguous() and 0<ids.numel()<=64)
        assert fused==expected_fused,'Source branch changed'
        assert local.dtype==torch.int32 and local.shape==ids.shape and local.device==ids.device
        assert weights.dtype==torch.float32 and weights.shape==ids.shape and weights.is_contiguous()
        assert weights.flatten().tolist()==[0.125]*len(values)
        if not fused:
            assert local.is_contiguous()
            assert local.flatten().tolist()==(values if mapping is None else reference(values,mapping))
        # In fused branch the source allocates local IDs for a later kernel.
        # Never inspect uninitialized values or claim that kernel was tested.
        return fused
