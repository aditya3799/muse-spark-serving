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

extern "C" int probe_count(void* opaque) {
    return int(static_cast<Probe*>(opaque)->algorithms.size());
}
extern "C" int probe_call(void* opaque, int index, const void* w, const void* x, void* y,
                          void* workspace, uint64_t bytes, cudaStream_t stream) {
    auto p = static_cast<Probe*>(opaque);
    if (index < 0 || size_t(index) >= p->algorithms.size()) return -1;
    float alpha = 1, beta = 0;
    return int(cublasLtMatmul(p->handle,p->op,&alpha,w,p->weight,x,p->input,&beta,
        y,p->output,y,p->output,&p->algorithms[index],workspace,bytes,stream));
}
extern "C" int probe_attr(void* opaque, int index, int field) {
    auto p = static_cast<Probe*>(opaque);
    const cublasLtMatmulAlgoConfigAttributes_t attrs[] = {
        CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_STAGES_ID};
    int value=-1;
    if(index<0 || size_t(index)>=p->algorithms.size() || field<0 || field>=5) return -1;
    auto status=cublasLtMatmulAlgoConfigGetAttribute(&p->algorithms[index],attrs[field],&value,sizeof(value),nullptr);
    return status == CUBLAS_STATUS_SUCCESS ? value : -1;
}
