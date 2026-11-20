"""Exercise the exact JIT predicate in host C++ plus physical ragged pack and causal oracle."""
import ctypes,hashlib,json,os,subprocess,tempfile
from pathlib import Path
import numpy as np
import torch
from flashinfer_variant import DECL
from flashinfer_pages import SharedPages

H=Path(__file__).resolve().parent
PRE=r"""
#include <cstdint>
#include <cmath>
#include <tuple>
#define __device__
namespace cute {using std::get;}
namespace flashinfer {
namespace math {constexpr float inf=INFINITY;}
struct StandardFP8Attention {template<class P,class C> StandardFP8Attention(const P&,const C&) {}};
}
#define REGISTER_LOGITS_TRANSFORM(p,s,b,q,k,qh,kh,...) template <class Params,class T> T LogitsTransform(const Params& p,T s,uint32_t b,uint32_t q,uint32_t k,uint32_t qh,uint32_t kh){ __VA_ARGS__ }
"""
POST=r"""
struct Extra {int64_t tree_left; const uint32_t* tree_descriptor;};
struct Params {Extra additional_params;};
template<int W,bool SW> void getmask(int id,int old,int left,const uint32_t* desc,uint8_t* out) {
  Params p{{left,desc}};auto coord=std::make_tuple(0,0,0,0,0,W,old+W,id);
  flashinfer::TreeFP8Attention<W,SW> v(p,coord);
  for(int q=0;q<W+2;++q)for(int k=0;k<old+W+2;++k)
    out[q*(old+W+2)+k]=std::isfinite(v.LogitsTransform(p,1.f,id,q,k,0,0));
}
extern "C" void check(int w,int sw,int b,int p,int left,const uint32_t* d,uint8_t* a) {
  if(w==8){if(sw)getmask<8,true>(b,p,left,d,a);else getmask<8,false>(b,p,left,d,a);}
  else if(w==12){if(sw)getmask<12,true>(b,p,left,d,a);else getmask<12,false>(b,p,left,d,a);}
  else {if(sw)getmask<16,true>(b,p,left,d,a);else getmask<16,false>(b,p,left,d,a);}
}
"""
def path(parent,j):
    seq=[j]
    while j>0:j=parent[j];seq.append(j)
    return seq[::-1]
def result():
    assert not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET")=="1"
    masks=0; outputs=0; checked_pages=0; sim_fp8=0; chain_eq=0
    rng=np.random.default_rng(1973067)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp); s=folder/"tree.cc"; so=folder/"tree.so"
        s.write_text(PRE+DECL.replace("#include <flashinfer/attention/hopper/variants.cuh>","")+POST)
        subprocess.run(["g++","-std=c++17","-O3","-fPIC","-shared",str(s),"-o",str(so)],check=True)
        fn=ctypes.CDLL(str(so)).check;fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*2
        for width in (8,12,16):
            for seed in range(9):
                batch=4; left=max(width-1,[18,64,2047][seed%3])
                pre=[0,width-1,left+3,2*left+5];cap=max(pre)+width
                req=torch.tensor([5,1,3,2]);lens=torch.tensor(pre,dtype=torch.int32)
                table=torch.from_numpy(rng.permutation((batch+2)*cap)).to(torch.int32).reshape(batch+2,cap)
                slots=table[req[:,None],lens[:,None]+torch.arange(width)].long()
                dm=[torch.from_numpy(rng.permutation(table.numel())).to(torch.int32) for _ in range(2)]
                parents=[];desc=[];depth=[]
                for b in range(batch):
                    pa=[-1]+[q-1 if (b==0 or seed%3==0) else int(rng.integers(q)) for q in range(1,width)]
                    parents.append(pa)
                    a=[path(pa,q) for q in range(width)]
                    desc.append([sum(1<<x for x in z) for z in a]);depth.append([len(z)-1 for z in a])
                depths=torch.tensor(depth,dtype=torch.int32);bits=torch.tensor(desc,dtype=torch.uint32)
                planner=SharedPages(batch,cap,width,left,"cpu")
                planner.stage(table,req,lens,slots,*dm,depths,bits)
                encoded=planner.descriptor.numpy();assert np.array_equal(encoded,(depths.numpy().astype(np.uint32)<<16)|bits.numpy())
                # Finite e4m3fn values are represented exactly as FP32 during the serial reference.
                k=torch.randn(table.numel(),4,generator=torch.Generator().manual_seed(seed+112)).to(torch.float8_e4m3fn).float().numpy()
                v=k[:,::-1].copy()
                for b in range(batch):
                    for sw in (False,True):
                        old=min(pre[b],left) if sw else pre[b]
                        indptr=(planner.swa_ptr if sw else planner.full_ptr).tolist()
                        physical=(planner.sliding if sw else planner.full)[indptr[b]:indptr[b+1]].numpy()
                        expect_ids=torch.cat((table[req[b],pre[b]-old:pre[b]],slots[b]))
                        want=dm[sw][expect_ids.long()].numpy()
                        assert np.array_equal(physical,want);checked_pages+=len(want)
                        gates=np.empty((width+2,old+width+2),dtype=np.uint8)
                        fn(width,sw,b,old,left,encoded.ctypes.data,gates.ctypes.data)
                        for q in range(width):
                            # The ordinary native causal scheduler additionally excludes future ordinals.
                            cols=np.flatnonzero(gates[q,:old+q+1])
                            lineage=path(parents[b],q)
                            old_start=max(0,old+len(lineage)-1-left) if sw else 0
                            wanted=np.r_[np.arange(old_start,old),old+np.array(lineage)]
                            assert np.array_equal(cols,wanted);masks+=1
                            q8=k[physical[old+q]]
                            def attn(ix):
                                z=(k[ix]*q8).sum(1)*.5; e=np.exp(z-z.max())
                                return (e[:,None]*v[ix]).sum(0)/e.sum()
                            assert np.array_equal(attn(physical[cols]),attn(want[wanted]));outputs+=1;sim_fp8+=len(wanted)
                            if len(lineage)==q+1:
                                dense=np.arange(max(0,old+q-left) if sw else 0,old+q+1)
                                assert np.array_equal(cols,dense);chain_eq+=1
                        # Check the padded query index path never indexes descriptor out-of-bounds.
                        assert gates[width:].all()
    meta=dict(scope="Exact shipped JIT transform body, compiled on CPU with its C++ constructor/index conventions; original Triton CSR layout in CPU interpreter; E4M3 inputs with FP32 serial attention. Hopper FP8 online-softmax numerics and device execution remain unchecked.",
        jit_source_sha256=hashlib.sha256(DECL.encode()).hexdigest(),widths=[8,12,16],variant_q_rows=masks,
        serial_attention_outputs=outputs,native_causal_or_swa_chain_rows=chain_eq,physical_page_entries=checked_pages,
        qk_pairs=sim_fp8)
    (H/"flashinfer_cpu.json").write_text(json.dumps(meta,indent=2));print(meta)
if __name__=="__main__":result()

