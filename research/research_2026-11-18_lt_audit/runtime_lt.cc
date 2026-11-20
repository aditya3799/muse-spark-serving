// Query runtime full-shape candidates. Accepted plans are legal for both layouts.
// Matrix descriptors use the column-major transpose identity for row-major F.linear.
#include <cublasLt.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

struct Choice {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t x{}, full_w{}, local_w{}, full_y{}, local_y{};
    cublasLtMatmulPreference_t pref{};
    struct Item { cublasLtMatmulAlgo_t algo; size_t local_bytes, full_bytes; int ordinal; };
    std::vector<Item> items;
    ~Choice() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        for (auto mat:{x,full_w,local_w,full_y,local_y}) if(mat) cublasLtMatrixLayoutDestroy(mat);
        if(op) cublasLtMatmulDescDestroy(op);
        if(handle) cublasLtDestroy(handle);
    }
};

extern "C" void* projection_prepare(int m,int n,int k,int world,uint64_t bytes) {
    if(m<=0 || k<=0 || world<=1 || n%world || n<world) return nullptr;
    auto p=std::make_unique<Choice>();
    auto mat=[](cublasLtMatrixLayout_t* desc,int rows,int cols) {
        return cublasLtMatrixLayoutCreate(desc,CUDA_R_16BF,rows,cols,rows);
    };
    cublasOperation_t trans=CUBLAS_OP_T;
    if (cublasLtCreate(&p->handle)
        || cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F)
        || cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans))
        || mat(&p->x,k,m) || mat(&p->full_w,k,n) || mat(&p->local_w,k,n/world)
        || mat(&p->full_y,n,m) || mat(&p->local_y,n/world,m)
        || cublasLtMatmulPreferenceCreate(&p->pref)
        || cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)))
        return nullptr;
    cublasLtMatmulHeuristicResult_t results[64]{};
    int count=0;
    if(cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,p->full_w,p->x,p->full_y,p->full_y,
                                     p->pref,64,results,&count)) return nullptr;
    for(int i=0;i<count;++i) {
        auto &a=results[i];
        cublasLtMatmulHeuristicResult_t local{};
        if(a.state || a.workspaceSize>bytes) continue;
        if(cublasLtMatmulAlgoCheck(p->handle,p->op,p->local_w,p->x,p->local_y,p->local_y,&a.algo,&local)
           || local.state || local.workspaceSize>bytes) continue;
        p->items.push_back({a.algo,local.workspaceSize,a.workspaceSize,i});
    }
    return p.release();
}
extern "C" uint64_t projection_lt_version(){return cublasLtGetVersion();}
extern "C" void projection_drop(void* q){delete static_cast<Choice*>(q);}
extern "C" int projection_count(void* q){return static_cast<Choice*>(q)->items.size();}
extern "C" uint64_t projection_attr(void* q,int i,int a) {
    auto &v=static_cast<Choice*>(q)->items.at(i);
    return a==0?v.ordinal:(a==1?v.local_bytes:v.full_bytes);
}
extern "C" int projection_call(void* q,int i,int full,const void* w,const void* x,void* y,
                              void* scratch,uint64_t bytes,cudaStream_t stream) {
    auto* p=static_cast<Choice*>(q);
    auto &choice=p->items.at(i);
    auto weight=full?p->full_w:p->local_w, out=full?p->full_y:p->local_y;
    if(bytes<(full?choice.full_bytes:choice.local_bytes)) return CUBLAS_STATUS_INVALID_VALUE;
    float alpha=1, beta=0;
    return cublasLtMatmul(p->handle,p->op,&alpha,w,weight,x,p->x,&beta,y,out,y,out,
                         &choice.algo,scratch,bytes,stream);
}

