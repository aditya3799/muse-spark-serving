"""Package the two serving expert implementations as genuine unscored MoE kernel tasks."""
import ast
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parent
parent = root / "release_2026-11-18"
native = root / "research_2026-11-18_target_moe/native_package_02"
assert hashlib.sha256((native/"candidate.diff").read_bytes()).hexdigest() == "1ce48266f47574c5c0d4fd3c557b12f6d24811d03121f9bd9ea4e89965f6065c"
assert hashlib.sha256((parent/"candidate.diff").read_bytes()).hexdigest() == "8c7547213fb993a318321a26f84e2872db8c87b358ee3d3ea638695702b0dc2b"
sg = parent/"candidate_sources/python/sglang"
ops = sg/"kernels/ops/moe"
header = "# okc: problem=muse_moe_w4a16\n"
common = (ops/"muse_spark_post_expert_finalize_triton.py").read_text() + '''
def extras(weights):
    scale = weights["w2_gs"][:32]
    norm = (weights["post_norm_w"].float() + 1).bfloat16()
    lookup = torch.full((257,), -1, device=norm.device, dtype=torch.int32)
    lookup[1:33] = torch.arange(32, device=norm.device, dtype=torch.int32)
    return norm, scale.bfloat16().contiguous(), lookup

def kernel_fn(data, state):
    cfg = data["config"]
    if (cfg["local_start"], cfg["local_count"]) != (0, 32):
        raise ValueError("expected the EP8 local expert bank")
    bank, norm, scale, lookup = state
    ids = lookup[data["topk_ids"].clamp_min(-1).to(torch.int32) + 1].contiguous()
    x = data["hidden_states"].contiguous()
    y = project(bank,x,ids)
    return fused_post_expert_norm_weighted_sum(y,data["topk_weights"],norm_weight=norm,
        weight_first=True,slot_expert_ids=ids,expert_output_scale=scale)
'''
cpp = """
std::vector<int64_t> configure();
void project_cuda(std::vector<torch::Tensor> tensors, int64_t grid);
"""
cuda = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
std::vector<int64_t> configure() {
  int grid;
  auto status = expert_config(&grid);
  TORCH_CHECK(status == 0, "Hopper configuration: ", status);
  size_t counts[3];
  expert_sizes(counts);
  return {grid,(int64_t)counts[0],(int64_t)counts[1],(int64_t)counts[2]};
}
void project_cuda(std::vector<torch::Tensor> a, int64_t grid) {
  int error = expert_run(
    (const unsigned char*)a[0].data_ptr(), (const unsigned char*)a[1].data_ptr(),
    (const float*)a[2].data_ptr(), (const uint16_t*)a[3].data_ptr(),
    (const __nv_bfloat16*)a[4].data_ptr(), (const int*)a[5].data_ptr(),
    (__nv_bfloat16*)a[6].data_ptr(), (int*)a[7].data_ptr(), (float*)a[8].data_ptr(),
    (__nv_bfloat16*)a[9].data_ptr(), a[4].size(0), a[2].numel(), grid, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(error == 0,"Hopper launch: ",error);
}
"""
def literal(text):
    assert "'''" not in text
    return "r'''\n"+text+"\n'''"
native_code = (native/"hopper_pack.py").read_text() + "\nCPP="+literal(cpp)+"\nCUDA="+literal((native/"hopper_expert.cu").read_text()+cuda)
native_code += '''
def prepare(weights):
    from torch.utils.cpp_extension import load_inline
    ext = load_inline(name="muse_hopper_experts_native_01",cpp_sources=CPP,cuda_sources=CUDA,
        functions=["configure","project_cuda"],extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a"],verbose=False)
    grid, counters, ready, tiles = ext.configure()
    w13 = pack_experts(weights["w13"][:32],weights["w13_sf"][:32],True)
    w2 = pack_experts(weights["w2"][:32],weights["w2_sf"][:32],False)
    gs = weights["w13_gs"][:32].bfloat16().float().contiguous()
    fp4 = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device=w2.device)
    fp8 = torch.arange(256,device=w2.device).byte().view(torch.float8_e4m3fn).float()
    table = (fp8[:,None]*fp4).bfloat16()
    return ((ext,grid,counters,ready,tiles,[w13,w2,gs,table]),*extras(weights))

def project(bank,x,ids):
    ext,grid,nc,nr,nt,weights = bank
    counters = torch.zeros(nc,device=x.device,dtype=torch.int32)
    ready = torch.zeros(nr,device=x.device,dtype=torch.int32)
    tiles = x.new_empty(nt)
    out = x.new_empty((x.size(0),8,4096))
    for lo in range(0,x.size(0),384):
        sl = slice(lo,lo+384)
        ext.project_cuda(weights+[x[sl],ids[sl],out[sl],counters,ready,tiles],grid)
    return out
'''
humming = (ops/"muse_spark_humming_scale.py").read_text()+"\n"+(ops/"muse_spark_humming_split.py").read_text()
path = sg/"srt/layers/muse_spark/muse_spark_nvfp4_moe.py"
source = path.read_text()
for node in ast.parse(source).body:
    if getattr(node,"name",None) in ("_prepare_humming_experts","_humming_expert_outputs_default","_humming_expert_outputs"):
        humming += "\n"+ast.get_source_segment(source,node)+"\n"
humming = "\n".join(l for l in humming.splitlines() if not l.lstrip().startswith(("from sglang.kernels.ops.moe.muse_spark_humming_scale import","from sglang.kernels.ops.moe.muse_spark_humming_split import")))
humming += '''
def prepare(weights):
    from types import SimpleNamespace
    raw = SimpleNamespace()
    for stage in ("w13","w2"):
        setattr(raw,stage+"_weight",weights[stage][:32])
        setattr(raw,stage+"_weight_scale",weights[stage+"_sf"][:32])
        setattr(raw,stage+"_weight_scale_2",weights[stage+"_gs"][:32])
    return (_prepare_humming_experts(raw,torch),*extras(weights))

def project(bank,x,ids):
    return _humming_expert_outputs(bank,x,ids,torch,defer_downscale=True,global_experts=256)[0]
'''
for name,body in (("expert_hopper.py",native_code),("expert_humming.py",humming)):
    data = (header+common+"\n"+body+"\n").encode()
    ast.parse(data)
    (here/name).write_bytes(data)
    print(name,len(data),hashlib.sha256(data).hexdigest())
