import ast, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch

OUT=Path(__file__).parent
SGL=OUT.parent.parent.parent/'sglang/python/sglang'
SOURCE=OUT/'candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
metadata_kernel=module(SGL/'kernels/ops/attention/metadata.py','isolated_metadata_kernel')
glue_module=module(SGL/'srt/model_executor/runner/metadata_glue_graph.py','isolated_metadata_glue')
names={'glue_draft_metadata','_set_decode_page_metadata','_apply_cuda_graph_metadata','_init_full_cg_decode_metadata','init_forward_metadata_out_graph'}
tree=ast.parse(SOURCE.read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
parallel=NS(attn_cp_size=1)
scope=dict(torch=torch,SpeculativeAlgorithm=NS(from_string=lambda s:NS(is_dflash=lambda:s=='DFLASH')),get_spec=lambda:NS(speculative_algorithm='DFLASH'),get_parallel=lambda:parallel,_should_disable_scheduler_metadata_precompute=lambda:False,normal_decode_set_metadata=metadata_kernel.normal_decode_set_metadata)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])),str(SOURCE),'exec'),scope)

class Mode:
    def is_extend(self):return False
    def is_decode_or_idle(self):return False
    def is_target_verify(self):return True
    def is_draft_extend_v2(self):return False
    def is_dllm_extend(self):return False

def run():
    b=scope['FlashAttentionBackend']()
    for attr,val in dict(is_draft_runner=True,fa_impl_ver=4,topk=0,speculative_num_draft_tokens=8,is_encoder_decoder=False,use_mla=False,local_attn_builder=None,use_sliding_window_kv_pool=False,_kv_shard_pool=None,needs_cpu_seq_lens=False,is_prefill_aware_swa=False).items():setattr(b,attr,val)
    b.kv_index_translator=NS(is_translating=False,reads_are_translated=False)
    assert b.glue_draft_metadata()==''
    guards=0
    for attr,val in [('is_draft_runner',False),('fa_impl_ver',3),('topk',2),('speculative_num_draft_tokens',None),('is_encoder_decoder',True),('use_mla',True),('local_attn_builder',object()),('use_sliding_window_kv_pool',True),('_kv_shard_pool',object()),('needs_cpu_seq_lens',True),('is_prefill_aware_swa',True)]:
        old=getattr(b,attr);setattr(b,attr,val);assert b.glue_draft_metadata();setattr(b,attr,old);guards+=1
    for attr in ('is_translating','reads_are_translated'):
        setattr(b.kv_index_translator,attr,True);assert b.glue_draft_metadata();setattr(b.kv_index_translator,attr,False);guards+=1
    parallel.attn_cp_size=2;assert b.glue_draft_metadata();parallel.attn_cp_size=1;guards+=1
    # Released Qwen3 DFlash2 uses sliding attention over an ordinary MHA
    # pool, not the separate SWAKVPool layout excluded by the opt-in.
    b.has_swa=True;b.sliding_window_size=2048
    b.page_size=1;b.max_num_pages=4096;b.max_context_len=4096
    b.req_to_token=torch.arange(16*4096,dtype=torch.int32,device='cuda').view(16,4096)
    b._bind_metadata_buffers=lambda *args:None
    b._capture_unified_verify_metadata=lambda fb:False
    b.attn_backend_list=None
    b.target_verify_metadata={}
    cases={}
    for bs in (1,4,8):
        m=NS(cache_seqlens_int32=torch.zeros(bs,dtype=torch.int32,device='cuda'),cu_seqlens_k=torch.zeros(bs+1,dtype=torch.int32,device='cuda'),page_table=torch.zeros(bs,4096,dtype=torch.int32,device='cuda'),swa_page_table=None)
        b.target_verify_metadata[bs]=m
        fb=NS(batch_size=bs,req_pool_indices=torch.arange(bs,dtype=torch.int32,device='cuda'),seq_lens=torch.full((bs,),17,dtype=torch.int32,device='cuda'),seq_lens_cpu=torch.full((bs,),999,dtype=torch.int32),encoder_lens=None,forward_mode=Mode(),spec_info=NS(),kv_loc_plan=object(),out_cache_loc=None)
        cases[bs]=(m,fb)
    glue=glue_module.MetadataGlueGraph(torch.device('cuda'))
    checks=0
    for iteration in range(8):
        for bs in (1,4,8,4,1):
            m,fb=cases[bs]
            req=(torch.arange(bs,device='cuda',dtype=torch.int32)+iteration+1)%16
            lens=((torch.arange(bs,device='cuda',dtype=torch.int32)*743+iteration*661)%4000)+1
            fb.req_pool_indices.copy_(req);fb.seq_lens.copy_(lens)
            # Real host plan objects change every step; guarded backend never
            # reads their tables. In-place pool mutations must remain visible.
            fb.kv_loc_plan=object();fb.seq_lens_cpu.fill_(9999-iteration)
            b.req_to_token.add_(3)
            glue.run(b,fb,('draft',bs))
            torch.cuda.synchronize()
            assert b.forward_metadata is m
            want=lens+8
            assert torch.equal(m.cache_seqlens_int32,want)
            assert torch.equal(m.cu_seqlens_k,torch.nn.functional.pad(want.cumsum(0,dtype=torch.int32),(1,0)))
            for row in range(bs):
                n=int(want[row]);assert torch.equal(m.page_table[row,:n],b.req_to_token[int(req[row]),:n])
            checks+=1
    assert glue.replays>0 and not glue.disabled
    return dict(passed=True,candidate_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),guard_cases=guards,metadata_cases=checks,glue_replays=glue.replays,keys=len(glue._states),device=torch.cuda.get_device_name(),limitations='Metadata equivalence only, exact exported backend methods and original MetadataGlueGraph/Triton metadata kernel; no draft model, acceptance, FA4 attention or serving throughput measured')
if __name__=='__main__':
    try:result=run()
    except Exception as exc:
        import traceback;traceback.print_exc();result=dict(passed=False,error=str(exc),exception_type=type(exc).__name__)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
