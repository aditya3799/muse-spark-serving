"""One request-index table and one native Hopper FP8 online-softmax per tree query.

Use with flashinfer-python==0.7.2. One packed uint32 = low 16 ancestor bits,
high 16 logical depth. The last W physical pages in each request represent its
ordinary ordinal verify reservation; all positions/RoPE and KV writes precede this call.
"""
DECL=r"""
#include <flashinfer/attention/hopper/variants.cuh>
namespace flashinfer {
template <int WIDTH, bool SLIDING>
struct TreeFP8Attention : StandardFP8Attention {
  int prefix;
  template <typename Params, typename Coord>
  __device__ TreeFP8Attention(const Params& p, const Coord& c) : StandardFP8Attention(p,c) {
    prefix=int(cute::get<6>(c))-int(cute::get<5>(c));
  }
  REGISTER_LOGITS_TRANSFORM(p, score, b, q, k, qh, kh, {
    if(q>=WIDTH) return score;  // padded GMMA lanes produce no output
    const int key=int(k);
    if(key<prefix) {
      if constexpr (!SLIDING) return score;
      const int oldest=prefix-p.additional_params.tree_left;
      if(key>=oldest+WIDTH-1) return score;
      if(key<oldest) return T(-math::inf);
      const unsigned descriptor=p.additional_params.tree_descriptor[b*WIDTH+q];
      return key>=oldest+int(descriptor>>16) ? score : T(-math::inf);
    }
    const unsigned node=unsigned(key-prefix);
    if(node>=WIDTH) return T(-math::inf);
    const unsigned descriptor=p.additional_params.tree_descriptor[b*WIDTH+q];
    return ((descriptor>>node)&1u) ? score : T(-math::inf);
  })
};
}  // namespace flashinfer
"""
TENSORS=["maybe_scale_q","maybe_scale_k","maybe_scale_v","tree_descriptor"]
TD=["float","float","float","uint32_t"]
SCALARS=["sm_scale","scale_q_scalar","scale_k_scalar","scale_v_scalar","tree_left"]
SD=["double","double","double","double","int64_t"]

def make_wrapper(workspace, *, width, sliding, layout="NHD", graph_buffers=None):
    import torch
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
    if not 2<=width<=16 or layout not in ("NHD","HND"):
        raise ValueError("Unsupported width or KV layout")
    kind="true" if sliding else "false"
    return BatchPrefillWithPagedKVCacheWrapper(workspace,layout,backend="fa3",
        jit_args=[f"okc_dflash_tree_fp8_w{width}_s{int(sliding)}_d64_v1",torch.float8_e4m3fn,
            torch.float8_e4m3fn,torch.bfloat16,torch.int32,64,64,TENSORS,TD,SCALARS,SD,
            f"flashinfer::TreeFP8Attention<{width},{kind}>",DECL],
        jit_kwargs={"fp8_enabled":True,"use_sliding_window":False},
        **({} if graph_buffers is None else graph_buffers))

def plan(wrapper,qo_indptr,kv_indptr,indices,last_len,*,q_heads,kv_heads):
    """Pass paged CSR entries once per request, all physical IDs in the layer's pool."""
    import torch
    return wrapper.plan(qo_indptr,kv_indptr,indices,last_len,q_heads,kv_heads,64,1,
        causal=True,window_left=-1,pos_encoding_mode="NONE",q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,o_data_type=torch.bfloat16)

def run(wrapper,q,cache,descriptor,*,width,softmax_scale,window_left=-1,
        scale_q=None,scale_k=None,scale_v=None,out=None):
    import torch
    if (q.dtype!=torch.float8_e4m3fn or q.shape[-1]!=64 or descriptor.dtype!=torch.uint32
        or not descriptor.is_contiguous() or descriptor.shape!=(q.shape[0]//width,width)
        or q.shape[0]%width or (window_left>=0 and window_left<width-1)):
        raise ValueError("Incompatible native FP8 input, tree descriptor or SWA window")
    # 0.7.2 run consumes ordered JIT args; scalar scales are fallback when tensor is absent.
    return wrapper.run(q,cache,scale_q,scale_k,scale_v,descriptor,float(softmax_scale),
                       1.,1.,1.,int(window_left),out=out)

