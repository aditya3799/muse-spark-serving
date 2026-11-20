// Runtime-discovered full-M cuBLASLt plans checked for an unchanged-K local-M call.
// No model shapes or algorithm IDs are embedded. Exact output qualification is
// performed by the caller against the actual native operation and weights.
#include <cublasLt.h>
#include <cstdint>
#include <vector>

struct Plan {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t w{},x{},y{},lx{},ly{};
  cublasLtMatmulPreference_t pref{};
  struct Choice {cublasLtMatmulAlgo_t algo; uint64_t full_ws,local_ws;};
  std::vector<Choice> choices;
  ~Plan() {
    if(pref)cublasLtMatmulPreferenceDestroy(pref);
    for(auto v:{w,x,y,lx,ly})if(v)cublasLtMatrixLayoutDestroy(v);
    if(op)cublasLtMatmulDescDestroy(op);
    if(handle)cublasLtDestroy(handle);
  }
};
#define REQUIRE(expr) if((expr)!=CUBLAS_STATUS_SUCCESS){delete p;return nullptr;}
extern "C" void* make_row_plan(int m,int local_m,int n,int k,int weight_transpose,uint64_t bytes,int count){
  auto* p=new Plan;
  REQUIRE(cublasLtCreate(&p->handle));
  REQUIRE(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t tr=weight_transpose?CUBLAS_OP_T:CUBLAS_OP_N;
  REQUIRE(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&tr,sizeof(tr)));
  REQUIRE(cublasLtMatrixLayoutCreate(&p->w,CUDA_R_16BF,weight_transpose?k:n,weight_transpose?n:k,weight_transpose?k:n));
  REQUIRE(cublasLtMatrixLayoutCreate(&p->x,CUDA_R_16BF,k,m,k));
  REQUIRE(cublasLtMatrixLayoutCreate(&p->y,CUDA_R_16BF,n,m,n));
  REQUIRE(cublasLtMatrixLayoutCreate(&p->lx,CUDA_R_16BF,k,local_m,k));
  REQUIRE(cublasLtMatrixLayoutCreate(&p->ly,CUDA_R_16BF,n,local_m,n));
  REQUIRE(cublasLtMatmulPreferenceCreate(&p->pref));
  REQUIRE(cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
  std::vector<cublasLtMatmulHeuristicResult_t> result(count);int got=0;
  REQUIRE(cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,p->w,p->x,p->y,p->y,p->pref,count,result.data(),&got));
  for(int i=0;i<got;++i){
    auto &r=result[i]; if(r.state)continue;
    cublasLtMatmulHeuristicResult_t local{};
    if(cublasLtMatmulAlgoCheck(p->handle,p->op,p->w,p->lx,p->ly,p->ly,&r.algo,&local)||local.state)continue;
    if(r.workspaceSize>bytes||local.workspaceSize>bytes)continue;
    p->choices.push_back({r.algo,r.workspaceSize,local.workspaceSize});
  }
  return p;
}
extern "C" int row_plan_count(void* v){return static_cast<Plan*>(v)->choices.size();}
extern "C" int row_plan_attr(void* v,int index,int field){
  static const cublasLtMatmulAlgoConfigAttributes_t fields[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
    CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};
  auto*p=static_cast<Plan*>(v);int answer=-1;
  if(index<0||index>=int(p->choices.size())||field<0||field>=9)return -1;
  if(cublasLtMatmulAlgoConfigGetAttribute(&p->choices[index].algo,fields[field],&answer,sizeof(answer),nullptr))return -1;
  return answer;
}
extern "C" int row_plan_call(void*v,int index,int full,const void*w,const void*x,void*y,void*workspace,uint64_t bytes,cudaStream_t stream){
  auto*p=static_cast<Plan*>(v);if(index<0||index>=int(p->choices.size()))return -1;
  float alpha=1,beta=0;auto ix=full?p->x:p->lx;auto iy=full?p->y:p->ly;
  return cublasLtMatmul(p->handle,p->op,&alpha,w,p->w,x,ix,&beta,y,iy,y,iy,&p->choices[index].algo,workspace,bytes,stream);
}
extern "C" void row_plan_destroy(void*v){delete static_cast<Plan*>(v);}
