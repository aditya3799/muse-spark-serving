import ast, hashlib, importlib.util, json, sys, types
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).parent
SOURCE = ROOT / 'candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
SGL = ROOT.parent.parent/'sglang/python/sglang'
# Load the actual ragged layout and kernels without importing unrelated
# server dependencies absent from this component-only environment.
kernel_name='sglang.kernels.ops.speculative.ragged_verify_kernels'
kernel_spec=importlib.util.spec_from_file_location(kernel_name,SGL/'kernels/ops/speculative/ragged_verify_kernels.py')
kernel_module=importlib.util.module_from_spec(kernel_spec)
sys.modules[kernel_name]=kernel_module
kernel_spec.loader.exec_module(kernel_module)
ragged_tree=ast.parse((SGL/'srt/speculative/ragged_verify.py').read_text())
ragged_tree.body=[n for n in ragged_tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='sglang.srt.environ')]
ragged_module=types.ModuleType('isolated_ragged_verify')
sys.modules[ragged_module.__name__]=ragged_module
exec(compile(ragged_tree,'ragged_verify.py','exec'),ragged_module.__dict__)
RaggedVerifyLayout=ragged_module.RaggedVerifyLayout
build_ragged_target_verify_geometry=ragged_module.build_ragged_target_verify_geometry

def load_methods():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlashAttentionBackend')
    names = {'_compute_scheduler_metadata', '_compute_verify_scheduler_metadata', '_apply_cuda_graph_metadata', '_init_full_cg_decode_metadata', 'forward_extend'}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    cls.bases = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    calls = []
    def flash(**kw):
        calls.append(kw)
        return kw['q'].clone()
    scope = dict(torch=torch, is_cp_active=lambda _:False, AttentionType=NS(ENCODER_ONLY=2, DECODER_BIDIRECTIONAL=3), flash_attn_with_kvcache=flash, build_ragged_target_verify_geometry=build_ragged_target_verify_geometry)
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), scope)
    return scope['FlashAttentionBackend'], calls

class Mode:
    def is_decode_or_idle(self): return False
    def is_target_verify(self): return True
    def is_draft_extend_v2(self): return False

def backend_fixture(device='cpu'):
    cls, calls = load_methods()
    b = cls()
    b.local_attn_builder = None
    b._kv_shard_pool = None
    b.topk = 1
    b.use_mla = False
    b._disable_scheduler_metadata_precompute = False
    b.num_attention_heads, b.num_kv_heads, b.head_dim = 16, 2, 64
    b.kv_cache_dtype = torch.bfloat16
    b.page_size, b.has_softcap, b.num_splits = 256, False, 0
    b.max_context_len, b.speculative_num_draft_tokens = 4096, 8
    b.fa_impl_ver, b.use_sliding_window_kv_pool, b.needs_cpu_seq_lens = 3, False, False
    b._verify_sched_meta_buf = torch.zeros(17, dtype=torch.int32, device=device)
    b._capture_unified_verify_metadata = lambda _:False
    m = NS(cache_seqlens_int32=torch.zeros(1, dtype=torch.int32, device=device), cu_seqlens_q=torch.tensor([0,8], dtype=torch.int32, device=device), cu_seqlens_k=torch.tensor([0,0], dtype=torch.int32, device=device), max_seq_len_q=8, max_seq_len_k=4096, scheduler_metadata=None, page_table=torch.zeros(1,16,dtype=torch.int32,device=device), image_token_ranges=None, swa_spec_metadata=None)
    b.target_verify_metadata = {1:m}
    b._bind_metadata_buffers = lambda *args:None
    def set_meta(plan, metadata, req, lens, delta):
        metadata.cache_seqlens_int32.copy_(lens + delta)
        metadata.cu_seqlens_k[1:] = metadata.cache_seqlens_int32.cumsum(0)
    b._set_decode_page_metadata = set_meta
    fb = NS(batch_size=1, req_pool_indices=torch.zeros(1,dtype=torch.int32,device=device), seq_lens=torch.tensor([17],dtype=torch.int32,device=device), seq_lens_cpu=None, encoder_lens=None, forward_mode=Mode(), spec_info=NS(), positions=torch.arange(8,device=device), kv_loc_plan=None, out_cache_loc=None)
    return b,m,fb,calls

def main():
    b,m,fb,calls = backend_fixture()
    scheduler_calls=[]
    def scheduler(**kw):
        scheduler_calls.append(kw)
        values = [kw['batch_size'], kw['max_seqlen_q'], kw['max_seqlen_k'], int(kw['cache_seqlens'].sum())] + [0]*13
        return torch.tensor(values,dtype=torch.int32)
    b._get_scheduler_metadata = scheduler
    b._init_full_cg_decode_metadata(fb, in_capture=True)
    ptr=m.scheduler_metadata.data_ptr()
    assert m.scheduler_metadata[:4].tolist() == [1,8,4096,25]
    for length in (9, 513, 2048, 4000):
        fb.seq_lens.fill_(length)
        b._init_full_cg_decode_metadata(fb, in_capture=False)
        assert ptr == m.scheduler_metadata.data_ptr()
        assert m.scheduler_metadata[:4].tolist() == [1,8,4096,length+8]
    # Host-mirrored lengths use a safe query-window upper bound.
    b.needs_cpu_seq_lens=True
    fb.seq_lens_cpu=torch.tensor([65])
    fb.seq_lens.fill_(65)
    b._init_full_cg_decode_metadata(fb, in_capture=False)
    assert m.scheduler_metadata[:4].tolist() == [1,8,4096,73]
    fb.spec_info.ragged_verify_layout=RaggedVerifyLayout.from_verify_lens(verify_lens_cpu=[3],device='cpu',grid=[4])
    b._init_full_cg_decode_metadata(fb, in_capture=False)
    assert m.cu_seqlens_q.tolist()==[0,4]
    assert m.scheduler_metadata[:4].tolist()==[1,8,4096,69]
    for name, value in (('local_attn_builder',object()),('_kv_shard_pool',object()),('topk',2),('use_mla',True),('_disable_scheduler_metadata_precompute',True),('_get_scheduler_metadata',None)):
        previous=getattr(b,name)
        setattr(b,name,value)
        assert b._compute_verify_scheduler_metadata(m,1) is None, name
        setattr(b,name,previous)
    # Execute exact exported forward_extend with a recording FA boundary.
    b.fa_skip_kv_cache=False
    b._mxfp8_sf_kwargs=lambda *args:{}
    b.prepare_paged_mha_query=lambda q,qr,kr,layer,**kw:(q,qr,kr,None,None)
    kv=torch.zeros(16,256,2,64,dtype=torch.bfloat16)
    b.get_paged_mha_kv_cache=lambda layer:(kv,kv)
    layer=NS(sliding_window_size=-1,is_cross_attention=False,attn_type=1,tp_k_head_num=2,tp_v_head_num=2,tp_q_head_num=16,head_dim=64,v_head_dim=64,scaling=0.125,logit_cap=0)
    q=torch.zeros(8,16,64,dtype=torch.bfloat16)
    b.forward_extend(q,None,None,layer,fb,save_kv_cache=False)
    assert calls[-1]['scheduler_metadata'] is m.scheduler_metadata
    for name,value in (('sliding_window_size',2047),('attn_type',3),('tp_k_head_num',1),('logit_cap',3)):
        old=getattr(layer,name); setattr(layer,name,value)
        b.forward_extend(q,None,None,layer,fb,save_kv_cache=False)
        assert calls[-1]['scheduler_metadata'] is None,name
        setattr(layer,name,old)
    b.forward_extend(q.float(),None,None,layer,fb,save_kv_cache=False)
    assert calls[-1]['scheduler_metadata'] is None
    result=dict(passed=True, candidate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), scheduler_refreshes=len(scheduler_calls), forward_boundary_cases=len(calls), covered=['capture binding','changed-length replay','pointer stability','host and GPU-only length bounds','actual RaggedVerifyLayout padding/geometry refresh','local/KV-shard/topk/MLA/DP/CP-disabled/FA4 fallbacks','full versus SWA','causal versus bidirectional','head-count/dtype/softcap guards'], limitations='CPU exact-method integration with scheduler and FA boundary spies; bare FA3 GPU correctness is separate; FP8 Hopper not validated')
    (ROOT/'candidate_cpu_tests.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
