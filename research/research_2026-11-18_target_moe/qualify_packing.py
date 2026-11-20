"""Qualify a single packed Hopper representation and the W13 serving rounding boundary."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parent.parent
DONOR=ROOT.parent/"muse_moe_w4a16/submission_v139_wgs_kperm3.py"


def pack_stage(codes, scales, stage, pack, positions):
    """Write bounded 256-output-row tiles directly into the final rank-local buffer."""
    experts, rows, packed_k = codes.shape
    if packed_k!=2048 or rows not in (4096,8192) or scales.shape!=(experts,rows,256):
        raise ValueError("Unexpected expert shape")
    if (stage,rows) not in (("w13",8192),("w2",4096)):
        raise ValueError("Unexpected stage")
    device=codes.device
    packed=torch.empty((experts,rows//256,256*(packed_k+256)),device=device,dtype=torch.uint8)
    for first in range(0,rows,256):
        logical=torch.arange(first,first+256,device=device)
        if stage=="w13":
            logical=(logical//16)*8+logical%8+((logical%16)//8)*(rows//2)
        for expert in range(experts):
            tile=pack(codes[expert],scales[expert],logical,positions,kperm3=True)
            packed[expert,first//256].copy_(tile)
    return packed


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    digest=hashlib.sha256(DONOR.read_bytes()).hexdigest()
    assert digest=="a24b8374aad49fbaa5e0cdeffb7f1fa64960d8321f343cf8b88bd97dfdd20c09"
    tree=ast.parse(DONOR.read_text())
    names={"_wgs_copy","_pack_codes_nat","_stored_pos_mul"}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==3
    ns={"torch":torch}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(DONOR),"exec"),ns)
    positions=ns["_stored_pos_mul"]()
    result=dict(donor=digest,device=torch.cuda.get_device_name(),stages=[])
    torch.manual_seed(151)
    for stage,rows in (("w13",8192),("w2",4096)):
        # Different bytes in every expert, including scale bit patterns. Packing uses no arithmetic on weights.
        codes=torch.randint(256,(2,rows,2048),device="cuda",dtype=torch.uint8)
        scales=torch.randint(256,(2,rows,256),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)
        initial=torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        physical=pack_stage(codes,scales,stage,ns["_wgs_copy"],positions)
        scratch_and_output=torch.cuda.max_memory_allocated()-initial
        if stage=="w13":
            chunks=torch.arange(4096,device="cuda").view(-1,8)
            rows_ref=torch.stack((chunks,chunks+4096),dim=1).reshape(-1)
        else:
            rows_ref=torch.arange(rows,device="cuda")
        for i in range(2):
            reference=ns["_wgs_copy"](codes[i],scales[i],rows_ref,positions,kperm3=True)
            assert torch.equal(physical[i].flatten(),reference)
            assert not torch.equal(physical[1-i].flatten(),reference)
        canonical_bytes=physical.numel()
        assert canonical_bytes==codes.numel()+scales.numel()
        # Changed-input and bit sensitivity: alter one code and one scale with independent locations.
        codes[0,rows-1,31] ^= 1
        scales.view(torch.uint8)[1,17,99] ^= 128
        changed=pack_stage(codes,scales,stage,ns["_wgs_copy"],positions)
        changed_bytes=(changed!=physical).reshape(2,-1).sum(dim=1).tolist()
        assert changed_bytes==[1,1]
        result["stages"].append(dict(stage=stage,experts=2,matched_bytes=canonical_bytes,
                                      peak_pack_bytes=scratch_and_output,
                                      transient_above_destination=scratch_and_output-canonical_bytes,
                                      changed_bytes_by_expert=changed_bytes))
        del codes,scales,physical,changed,reference
    # Exercise the current serving JIT on BF16 GEMM outputs after their BF16 global scale.
    sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
    from kernel_runtime import bootstrap
    bootstrap(ROOT/"sglang")
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from importlib.util import spec_from_file_location, module_from_spec
    path=ROOT/"release_2026-11-18/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_scale.py"
    spec=spec_from_file_location("scored_scale",path)
    scale=module_from_spec(spec)
    spec.loader.exec_module(scale)
    accum=torch.randn((64,8192),device="cuda",dtype=torch.float32)
    global_scales=torch.linspace(.125,1.5,32,device="cuda").bfloat16()
    ids=(torch.arange(64,device="cuda",dtype=torch.int32)%32).reshape(8,8)
    effective=global_scales[ids.flatten().long()].float().unsqueeze(-1)
    reference=accum.bfloat16()
    scale.scale_active_rows(reference,ids,global_scales)
    expected=torch.empty((64,4096),device="cuda",dtype=torch.bfloat16)
    silu_and_mul(reference,expected,expert_ids=ids.flatten(),expert_step=1)
    def native_activate(pair):
        out=torch.empty_like(expected)
        silu_and_mul(pair,out,expert_ids=ids.flatten(),expert_step=1)
        return out
    rounded=(accum.bfloat16().float()*effective).bfloat16()
    assert torch.equal(rounded,reference)
    assert torch.equal(native_activate(rounded),expected)
    gate,up=(accum*effective).chunk(2,1)
    late=(torch.nn.functional.silu(gate)*up).bfloat16()
    result["rounding"]=dict(elements=expected.numel(),late_round_disagreement=int((late!=expected).sum()),
                             stock_round_disagreement=int((native_activate(rounded)!=expected).sum()),
                             scope="identical FP32 accumulators; late-round ordering emulation uses PyTorch SiLU, not a native donor execution")
    assert result["rounding"]["late_round_disagreement"]>0
    result["scope"]="Packing byte equality only. The donor WGMMA consumer requires SM90. Multi-rank execution and a common prefill consumer are open."
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()

