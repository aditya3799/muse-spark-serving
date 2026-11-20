"""Execute pinned upstream attention dispatch heuristics as a CPU-only program."""
import hashlib,json,pathlib,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[3];OUT=pathlib.Path(__file__).resolve().parent;P=OUT/'fa3_policy'
def extract(path,name):
 text=path.read_text();match=re.search(r'(?:inline|constexpr)\s+[^;{}]*?\b'+name+r'\s*\(',text)
 assert match,name
 begin=text.index('{',match.start());depth=1;end=begin+1
 while depth:
  if text[end]=='{':depth+=1
  elif text[end]=='}':depth-=1
  end+=1
 return text[match.start():end]
names=[('heuristics.h','use_one_mma_wg'),('heuristics.h','should_pack_gqa'),('heuristics.h','num_splits_heuristic'),('tile_size.h','tile_size_fwd_sm90'),('tile_size.h','tile_size_fwd_sm8x'),('flash_api.cpp','get_pagedkv_tma'),('flash_api.cpp','get_pack_gqa'),('flash_api.cpp','get_num_splits')]
header='''#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>
#include <iostream>
struct Flash_fwd_params {
 int arch=90,d=64,dv=64,d_rounded=64,dv_rounded=64,b=1,h=16,h_k=2,seqlen_q=8,seqlen_k=131072,total_q=8,num_sm=132,page_size=1,window_size_left=-1,window_size_right=0,num_splits=0;
 bool is_causal=true,is_local=false,is_e4m3=true,pagedkv_tma=false;float softcap=0;
 void *page_table=(void*)1,*leftpad_k=nullptr,*knew_ptr=nullptr,*cu_seqlens_q=(void*)1,*cu_seqlens_k=nullptr,*seqused_q=nullptr,*seqused_k=(void*)1;
};
'''
main='''
int main(){
 for(int dim:{64,128})for(int bs:{1,4,16,32})for(int q:{1,8,16,32})for(int context:{2048,8192,74160,131072})for(int page:{1,128})for(int local:{0,1}){
  Flash_fwd_params p;p.d=p.dv=p.d_rounded=p.dv_rounded=dim;p.b=bs;p.seqlen_q=q;p.total_q=bs*q;p.seqlen_k=context;p.page_size=page;p.is_local=local;p.is_causal=!local;p.window_size_left=local?2047:-1;
  p.pagedkv_tma=get_pagedkv_tma(p);p.num_splits=get_num_splits(p);
  auto tile=tile_size_fwd_sm90(dim,dim,p.is_causal,p.is_local,1,false,!p.pagedkv_tma,false,use_one_mma_wg(p));
  std::cout<<dim<<","<<bs<<","<<q<<","<<context<<","<<page<<","<<local<<","<<std::get<0>(tile)<<","<<std::get<1>(tile)<<","<<get_pack_gqa(p)<<","<<p.pagedkv_tma<<","<<p.num_splits<<","<<use_one_mma_wg(p)<<"\\n";
 }
}
'''
code=header+'\n\n'.join(extract(P/f,n) for f,n in names)+main
source=P/'policy_cpu.cpp';binary=P/'policy_cpu';source.write_text(code)
subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(binary)],check=True,timeout=60)
run=subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=10)
keys=['head_dim','batch','query_rows','context_bound','page_size','local_window2048','block_m','block_n','pack_gqa','paged_tma','allocated_num_splits','one_mma_warpgroup']
rows=[dict(zip(keys,map(int,line.split(',')))) for line in run.stdout.splitlines()]
assert len(rows)==512
report={'status':'PINNED_CPP_HEURISTIC_EXECUTION_NOT_H200_RUNTIME_PROFILE','source_commit':'f89bc2306632d1ec5f97b014dded4254f5b4a907','source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.suffix in ('.h','.cpp','.cu')},'original_functions_executed':[n for f,n in names],'assumptions':{'arch':90,'num_sms':132,'q_heads':16,'kv_heads':2,'kv_dtype':'FP8e4m3','varlen_q':True,'page_table':True,'head_dim64':'Target constructor defaults, checkpoint config not authenticated','head_dim128':'Generic comparison, not actual draft FP8 configuration'},'rows':rows,'findings':['PackGQA is already enabled by pinned dispatch for all tested paged/varlen shapes; enabling it is not a fresh optimization','At query8 and GQA8, one MMA warpgroup is already selected; effective packedQ=64','allocated_num_splits is the host allocation/dispatch upper bound; device preparation can choose per-request actual split counts','FA4 CuTe interface in pinned SGLang rejects FP8 on SM90; changing target attention backend is not a validated flag-only gain'],'limitations':['No GPU kernels, timing, checkpoint weights or runtime capture state measured','Different real model head geometry invalidates assumed geometry','Changing SplitKV changes floating-point reduction boundaries and needs numerical validation'],'references':['https://github.com/sgl-project/sgl-attn/tree/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper']}
(OUT/'FA3_POLICY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'all_pack_gqa':all(r['pack_gqa'] for r in rows),'verify8_full_context':[r for r in rows if r['head_dim']==64 and r['query_rows']==8 and r['context_bound']==131072 and r['page_size']==1 and not r['local_window2048']]}))
