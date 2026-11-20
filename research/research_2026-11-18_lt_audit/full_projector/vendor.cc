// Diagnostic library. Each configuration is proposed on this host by cuBLASLt.
// Search native BF16 matmul choices at fixed full K. The caller checks every output.
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Probe {
    cublasLtHandle_t handle{};
    cublasLtMatmulDesc_t op{};
    cublasLtMatrixLayout_t weight{}, input{}, output{}, parent_weight{}, parent_output{};
    cublasLtMatmulPreference_t pref{};
    std::vector<cublasLtMatmulAlgo_t> algorithms;
    ~Probe() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        for (auto layout : {weight, input, output, parent_weight, parent_output})
            if (layout) cublasLtMatrixLayoutDestroy(layout);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }
};

extern "C" void destroy_probe(void* opaque) { delete static_cast<Probe*>(opaque); }
#define CHECK(expr) do { auto code = (expr); if (code != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLASLt status %d at line %d\n", int(code), __LINE__); delete p; return nullptr; } } while (0)

extern "C" void* prepare_probe(int m, int n, int full_n, int k, uint64_t workspace_bytes) {
    auto* p = new Probe;
    CHECK(cublasLtCreate(&p->handle));
    CHECK(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    CHECK(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK(cublasLtMatrixLayoutCreate(&p->weight, CUDA_R_16BF, k, n, k));
    CHECK(cublasLtMatrixLayoutCreate(&p->input, CUDA_R_16BF, k, m, k));
    CHECK(cublasLtMatrixLayoutCreate(&p->output, CUDA_R_16BF, n, m, n));
    CHECK(cublasLtMatrixLayoutCreate(&p->parent_weight, CUDA_R_16BF, k, full_n, k));
    CHECK(cublasLtMatrixLayoutCreate(&p->parent_output, CUDA_R_16BF, full_n, m, full_n));
    CHECK(cublasLtMatmulPreferenceCreate(&p->pref));
    CHECK(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
    for (bool full : {true,false}) {
        auto w = full ? p->parent_weight : p->weight;
        auto y = full ? p->parent_output : p->output;
        cublasLtMatmulHeuristicResult_t results[32]{};
        int count = 0;
        CHECK(cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,w,p->input,y,y,p->pref,32,results,&count));
        for (int i=0; i<count; ++i) {
            if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
            bool duplicate = false;
            for (const auto& previous : p->algorithms)
                if (std::memcmp(&previous,&results[i].algo,sizeof(previous)) == 0) duplicate = true;
            if (!duplicate) p->algorithms.push_back(results[i].algo);
        }
    }
    return p;
}


#include <algorithm>
#include <array>
#include <set>
#include <tuple>
#include <memory>

static const cublasLtMatmulAlgoConfigAttributes_t fields[] = {
    CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
    CUBLASLT_ALGO_CONFIG_STAGES_ID, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
    CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID};

struct Entry {
    cublasLtMatmulAlgo_t algo;
    int source;
    uint64_t discovery_workspace;
    uint64_t parent_workspace;
    uint64_t local_workspace;
};
using Fields=std::array<int,9>;
static Fields describe(const cublasLtMatmulAlgo_t &algo) {
    Fields f{};
    for (int i=0;i<9;++i) {
        int value=-1;
        auto s=cublasLtMatmulAlgoConfigGetAttribute(&algo,fields[i],&value,sizeof(value),nullptr);
        f[i]=s==CUBLAS_STATUS_SUCCESS?value:-1;
    }
    return f;
}

struct Sweep {
    std::unique_ptr<Probe> p;
    std::vector<Entry> entries;
    std::set<Fields> seen;
    uint64_t cap;
    Sweep(Probe* state,uint64_t bytes):p(state),cap(bytes) {}
    void add(const cublasLtMatmulAlgo_t &algo,int from,uint64_t ws) {
        if (!seen.insert(describe(algo)).second) return;
        cublasLtMatmulHeuristicResult_t full{},local{};
        auto chk=[&](bool parent,cublasLtMatmulHeuristicResult_t *out) {
            auto w=parent?p->parent_weight:p->weight, y=parent?p->parent_output:p->output;
            return cublasLtMatmulAlgoCheck(p->handle,p->op,w,p->input,y,y,&algo,out);
        };
        auto sf=chk(true,&full),ss=chk(false,&local);
        if (sf||ss||full.state||local.state||full.workspaceSize>cap||local.workspaceSize>cap) return;
        entries.push_back({algo,from,ws,full.workspaceSize,local.workspaceSize});
    }
};

extern "C" void *make_sweep(int m,int n,int full_n,int k,int ldout,uint64_t native_ws,uint64_t cap,int max_split) {
    auto probe=static_cast<Probe*>(prepare_probe(m,n,full_n,k,cap));
    if (!probe) return nullptr;
    auto wrapper=std::make_unique<Sweep>(probe,cap);
    auto* p=wrapper->p.get();
    cublasLtMatrixLayoutDestroy(p->output);
    p->output=nullptr;
    if (cublasLtMatrixLayoutCreate(&p->output,CUDA_R_16BF,n,m,ldout)) return nullptr;
    std::vector<uint64_t> workspaces{native_ws,cap,0,128*1024,512*1024,1024*1024,
                                    2*1024*1024,4*1024*1024,8*1024*1024,16*1024*1024,32*1024*1024};
    for(auto bytes:workspaces) {
        if(bytes>cap) continue;
        cublasLtMatmulPreferenceSetAttribute(p->pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes));
        for(bool full:{true,false}) {
            auto w=full?p->parent_weight:p->weight, y=full?p->parent_output:p->output;
            cublasLtMatmulHeuristicResult_t results[64]{};
            int count=0;
            auto status=cublasLtMatmulAlgoGetHeuristic(p->handle,p->op,w,p->input,y,y,p->pref,64,results,&count);
            if(status) continue;
            for(int i=0;i<count;++i) if(!results[i].state) wrapper->add(results[i].algo,full?0:1,bytes);
        }
    }
    // Keep exact valid seed tiling, stages, swizzle and custom settings.
    // Sweep only permitted split-K counts and reduction modes, checking both
    // layouts with the installed CUDA version. Parameters are runtime input.
    auto seeds=wrapper->entries;
    if (max_split < 0 && !seeds.empty()) {
        // Anchor only to the native full-layout heuristic's reduction choices.
        // Tiles, kernel implementations and swizzle stay runtime-discovered.
        const auto anchor = describe(seeds.front().algo);
        const int count = anchor[2], reduction = anchor[3];
        for (const auto& seed : seeds) {
            auto a = seed.algo;
            if (cublasLtMatmulAlgoConfigSetAttribute(&a,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&count,sizeof(count))) continue;
            if (cublasLtMatmulAlgoConfigSetAttribute(&a,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction))) continue;
            wrapper->add(a,3,seed.discovery_workspace);
        }
        auto &entries = wrapper->entries;
        entries.erase(std::remove_if(entries.begin(),entries.end(),[&](const Entry& e) {
            const auto fields=describe(e.algo);
            return fields[2]!=count || fields[3]!=reduction;
        }),entries.end());
        return wrapper.release();
    }
    for(int split=1;split<=max_split;++split) {
        for(auto &seed:seeds) {
            auto a=seed.algo;
            if(cublasLtMatmulAlgoConfigSetAttribute(&a,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split))) continue;
            for(int red:{0,1,2,4}) {
                if(cublasLtMatmulAlgoConfigSetAttribute(&a,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&red,sizeof(red))) continue;
                wrapper->add(a,2,seed.discovery_workspace);
            }
        }
    }
    return wrapper.release();
}
extern "C" int sweep_count(void* q){return static_cast<Sweep*>(q)->entries.size();}
extern "C" void sweep_attr(void* q,int i,int* sink,uint64_t* work){
    const auto &e=static_cast<Sweep*>(q)->entries.at(i);
    auto f=describe(e.algo); std::copy(f.begin(),f.end(),sink); sink[9]=e.source;
    work[0]=e.discovery_workspace; work[1]=e.parent_workspace; work[2]=e.local_workspace;
}
extern "C" int sweep_call(void* q,int i,int full,const void* w,const void* x,void* y,
    void* scratch,uint64_t bytes,cudaStream_t stream) {
    auto* s=static_cast<Sweep*>(q);auto* p=s->p.get();const auto& a=s->entries.at(i).algo;
    auto weight=full?p->parent_weight:p->weight, output=full?p->parent_output:p->output;
    float alpha=1,beta=0;
    return cublasLtMatmul(p->handle,p->op,&alpha,w,weight,x,p->input,&beta,y,output,y,output,
                        &a,scratch,bytes,stream);
}
extern "C" void drop_sweep(void* q){delete static_cast<Sweep*>(q);}

