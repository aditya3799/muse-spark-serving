"""Enumerate NVIDIA's actual CuTe reordering atom on CPU and compare every nibble to native Humming."""
from pathlib import Path
import tempfile
import sys
import json
import subprocess
import hashlib
import compile_fp8 as cfg

d=Path(__file__).resolve().parent
inc=Path("/tmp/okc-cutlass-cd9d8a7d/include")
raw=(d/"reference/upstream_mixed_dtype_utils.hpp").read_text()
mid=raw.index("constexpr auto compute_memory_reordering_atom")
start=raw.rindex("template <",0,mid)
stop=raw.index("\n}",mid)+2
body=raw[start:stop]
src="""
#include <cstdio>
#include "cute/tensor.hpp"
#include "cute/arch/mma_sm90.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
namespace cutlass {
"""+body+"""
}
using namespace cute;
int humming(int n,int k,int N,bool wg){
  int lane=n%8*4+k%8/2;
  int p=4*(k%2)+(wg?2*((k/8)%2)+(n/8)%2:2*((n/8)%2)+(k/8)%2);
  return 8*((k/16)*(N*2)+(n/64)*128+lane*4+(n%64)/16)+p;
}
template<int N,int K,class Atom>
void check(char const* name, Atom atom){
  auto total=tile_to_shape(atom,make_shape(Int<N>{},Int<K>{},_1{}));
  int match_w=0,match_h=0,example[4]={-1,-1,-1,-1};
  for(int n=0;n<N;n++)for(int k=0;k<K;k++){
    int pos=total(n,k,0),w=humming(n,k,N,true),h=humming(n,k,N,false);
    match_w+=pos==w;match_h+=pos==h;
    if(pos!=w && example[0]<0){example[0]=n;example[1]=k;example[2]=pos;example[3]=w;}
  }
  std::printf("%s %d %d %d %d %d %d %d %d\\n",name,N,K,match_w,match_h,example[0],example[1],example[2],example[3]);
  print(atom);std::printf("\\n");
}
int main(){
 using Shuffle=Layout<Shape<_2,_4>,Stride<_4,_1>>;
 auto one=cutlass::compute_memory_reordering_atom<cutlass::bfloat16_t, Layout<Shape<_1,_1>>,Shuffle>();
 auto fourN=cutlass::compute_memory_reordering_atom<cutlass::bfloat16_t, Layout<Shape<_4,_1>>,Shuffle>();
 auto fourK=cutlass::compute_memory_reordering_atom<cutlass::bfloat16_t, Layout<Shape<_1,_4>>,Shuffle>();
 check<128,64>("one",one);check<128,64>("fourN",fourN);check<128,64>("fourK",fourK);
 check<256,128>("fourN",fourN);
}
"""
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp);cu=tmp/"test.cu";cu.write_text(src)
    cli=[str(cfg.nvcc),"-std=c++17","--expt-relaxed-constexpr","-O1","-cudart=none",f"-I{inc}",
         *(f"-I{x}" for x in cfg.incs),str(cu),"-o",str(tmp/"test")]
    lib=next(p for p in Path(sys.prefix).rglob("libcudart.so.*") if p.is_file())
    cli.extend([f"-L{lib.parent}",f"-l:{lib.name}",f"-Xlinker=-rpath={lib.parent}"])
    rr=subprocess.run(cli,capture_output=True,text=True)
    if rr.returncode:
        print(rr.stderr[:1900])
        raise SystemExit(rr.returncode)
    out=subprocess.check_output([str(tmp/"test")],text=True).splitlines()
recs=[]
for i in range(0,len(out),2):
    name,n,k,w,h,nn,kk,actual,expected=out[i].split()
    recs.append(dict(atom=name,N=int(n),K=int(k),same_wgmma=int(w),same_hmma=int(h),
                     first_wgmma_difference=list(map(int,(nn,kk,actual,expected))),cute_layout=out[i+1]))
(d/"atom_addresses.json").write_text(json.dumps(dict(upstream_utility_sha256=hashlib.sha256(raw.encode()).hexdigest(),cases=recs),indent=2))
print(recs)
