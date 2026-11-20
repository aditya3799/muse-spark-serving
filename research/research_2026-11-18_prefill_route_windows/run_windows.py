"""GPU AB/BA experiment with pinned SGLang alignment, same weights and both actual projections."""
from pathlib import Path
import argparse,ctypes,json,os,random,statistics,sys
import torch
from route_bounds import PACKAGE,ROOT,scenarios,summarize,plan

D=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"optimization_marlin_tiles_2026-11-15"))
sys.path.insert(0,str(PACKAGE.parent))
from kernel_runtime import bootstrap
bootstrap(ROOT/"sglang")
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
from native_package_02.hopper_pack import pack_experts
lib=ctypes.CDLL(str(D/"paired_windows.so")).paired
lib.argtypes=[ctypes.c_void_p]*9+[ctypes.c_int]*2+[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--sizes",type=int,nargs="+",default=[753,3012])
    parser.add_argument("--reps",type=int,default=4)
    parser.add_argument("--output",default="gpu_results.json")
    a=parser.parse_args()
    torch.manual_seed(1941)
    weights=[]
    for gate,n in ((True,8192),(False,4096)):
        codes=torch.randint(256,(32,n,2048),device="cuda",dtype=torch.uint8)
        block_scales=torch.randint(40,68,(32,n,256),device="cuda",dtype=torch.uint8).view(torch.float8_e4m3fn)
        weights.append(pack_experts(codes,block_scales,gate))
        del codes,block_scales
    gs=torch.linspace(.125,.5,32,device="cuda").bfloat16().float()
    fp4=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device="cuda")
    table=(torch.arange(256,device="cuda").byte().view(torch.float8_e4m3fn).float()[:,None]*fp4).bfloat16()
    tiles=torch.zeros(30932992,device="cuda",dtype=torch.bfloat16)
    ready=torch.zeros(608,device="cuda",dtype=torch.int32)
    stream=torch.cuda.current_stream().cuda_stream
    cases=[]
    def save():
        data=dict(device=torch.cuda.get_device_name(),reps=a.reps,cases=cases,
            scope="Paired SM86 warp-MMA functional and timing proxy. Both paths share the same substitution for Hopper TMA and WGMMA. Pinned SGLang sort per call included. CUDA event wall time in ms. Inputs are synthetic; actual serving routing unavailable.")
        (D/a.output).write_text(json.dumps(data,indent=2))
    for pattern,m,seed,orig in scenarios():
        if m not in a.sizes or seed not in (0,12) or pattern not in ("kernel_track_donor_fixture","donor_histogram_shuffled","uniform_global_top8"): continue
        ids=torch.tensor(orig,device="cuda",dtype=torch.int32)
        x=torch.randn((m,4096),device="cuda",dtype=torch.bfloat16)*.023
        out0=x.new_empty((m,8,4096))
        out1=torch.empty_like(out0)
        def execute(glob):
            if glob:
                sorted_slots,expert_map,layout=moe_align_block_size(ids,64,32,return_inactive_prefix_count=True)
                max_windows=(expert_map.numel()+47)//48
                desc=[t.data_ptr() for t in (sorted_slots,expert_map,layout)]
            else:
                max_windows=(m+383)//384
                desc=[None,None,None]
            for w in range(max_windows):
                first=0 if glob else w*384
                count=m if glob else min(384,m-first)
                output=out1 if glob else out0
                tensors=(*weights,gs,table,x[first],ids[first],output[first],tiles,ready)
                rc=lib(*(t.data_ptr() for t in tensors),count,32,*desc,w,stream)
                if rc: raise RuntimeError(rc)
        execute(False);execute(True)
        torch.cuda.synchronize()
        case=dict(pattern=pattern,m=m,seed=seed,ms_reference=[],ms_windows=[])
        # Examine active values in bounded chunks to cap temporary allocations.
        dl=bl=match=total=0
        for i in range(0,m,128):
            live=ids[i:i+128]>=0
            v=out0[i:i+128][live].float(); u=out1[i:i+128][live].float()
            dl+=float(((u-v)**2).sum());bl+=float((v**2).sum())
            match+=int((u==v).sum());total+=v.numel()
        case.update(live_values=total,relative_l2=(dl/bl)**.5 if bl else 0,exact=match/total if total else 1)
        cases.append(case);save()
        assert case["relative_l2"]<.01,case
        for rep in range(a.reps):
            for glob in ([False,True] if rep%2==0 else [True,False]):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();execute(glob);end.record();end.synchronize()
                case["ms_windows" if glob else "ms_reference"].append(start.elapsed_time(end))
        case["ratio_medians"]=statistics.median(case["ms_reference"])/statistics.median(case["ms_windows"])
        save()
        print(pattern,m,case["exact"],case["relative_l2"],case["ratio_medians"],flush=True)
if __name__=="__main__":main()
