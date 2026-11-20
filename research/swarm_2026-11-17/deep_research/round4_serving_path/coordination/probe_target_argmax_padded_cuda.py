"""CUDA integration gate for inert argmax overlay, not a full model test.
Actual FullCudaGraphBackend methods and overlaid capture lifecycle execute.
TP8 is emulated on one GPU; allocator-scope/barrier helpers are stubs.
No NCCL, timing, H200, complete runner, or serving qualification.
"""
from __future__ import annotations
import argparse, ast, contextlib, hashlib, importlib.util, json, time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Callable, Dict, Optional
import torch

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'RULES.md').exists())
SRT=ROOT/'sglang/python/sglang/srt'
OVER=HERE.parent/'expert_partition/target_argmax_overlay_v2/python/sglang/srt'

def extract(path,ns,names=None,cls=None):
    tree=ast.parse(path.read_text())
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    selected=[n for n in body if not isinstance(n,(ast.Import,ast.ImportFrom)) and (names is None or getattr(n,'name',None) in names)]
    tree=ast.Module(body=ast.parse('from __future__ import annotations').body+selected,type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)

def main(out):
    start=time.monotonic();out=out/'probe';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(8821)
    assert torch.cuda.mem_get_info()[0] >= (512+128)*2**20
    torch.cuda.reset_peak_memory_stats()
    checks=[]
    def ok(name,value):
        assert value,name
        checks.append(name)
    softpath=ROOT/'sglang/python/sglang/kernels/ops/sampling/softcap_row_argmax.py'
    spec=importlib.util.spec_from_file_location('pinned_softcap',softpath)
    soft=importlib.util.module_from_spec(spec);spec.loader.exec_module(soft)
    class Group:
        rank_in_group=0;world_size=8;ranks=list(range(8))
        def barrier(self): pass
        def all_gather_into_tensor(self,output,local):
            # Explicit local emulation; communication performance is not measured.
            n=local.shape[0];view=output.view(8,n,2);view[0].copy_(local)
            for rank in range(1,8):
                view[rank].copy_(soft.softcap_row_argmax(self.logits[rank,:n],20.0,25256,rank*25256))
    group=Group()
    group.logits=torch.randn(8,80,25256,dtype=torch.float32,device='cuda')
    parallel=NS(tp_size=8,tp_group=group,attn_dp_enabled=False,attn_cp_size=1,attn_dcp_size=1,pp_size=1)
    ns=dict(torch=torch,get_parallel=lambda:parallel,softcap_row_argmax=soft.softcap_row_argmax)
    extract(SRT/'layers/logits_processor.py',ns,['deferred_argmax'],'LogitsProcessor')
    class LP:
        final_logit_softcapping=20.0;vocab_size=202048;defer_verify_gather=True
        deferred_argmax=ns['deferred_argmax']
        def is_deferred_shard(self,x):return x is not None and x.shape[-1]<self.vocab_size
    lp=LP()
    extract(OVER/'speculative/dflash_target_argmax_capture.py',ns)
    ns.update(get_disagg=lambda:NS(enable_pdmux=False),get_exec=lambda:NS(overlap=NS(enable_two_batch_overlap=False)))
    extract(OVER/'speculative/dflash_target_argmax_adapter.py',ns)
    class Precarve:
        def measure(self):return contextlib.nullcontext()
        def mint(self):pass
    backend_ns=dict(torch=torch,contextmanager=contextlib.contextmanager,BaseCudaGraphBackend=object,
                    GraphPoolPrecarve=Precarve,get_parallel=lambda:parallel,
                    TorchMemorySaverAdapter=NS(create=lambda **kw:None),get_bool_env_var=lambda key:False,
                    get_or_create_global_graph_memory_pool=lambda module:torch.cuda.graph_pool_handle(),
                    set_graph_pool_id=lambda pool:None,graph_pool_capture_scope=contextlib.nullcontext,
                    graph_pool_replay_scope=contextlib.nullcontext)
    backend_path=SRT/'model_executor/runner_backend/full_cuda_graph_backend.py'
    extract(backend_path,backend_ns,['FullCudaGraphBackend'])
    @dataclass(frozen=True)
    class Key:
        size:int
        stream_idx:object=None
        variant_label:object=None
        attention_variant:object=None
    mode=NS(is_target_verify=lambda:True)
    worker=type('DFlashWorkerV2',(),{})();worker.device='cuda';worker.block_size=8;worker._is_domino=False
    worker._target_logits_processor=lp
    mr=NS(is_draft_worker=False,model_config=NS(is_encoder_decoder=False),lora_manager=None,capture_tail_hooks=[],forward_stream=torch.cuda.current_stream())
    worker.target_worker=NS(model_runner=mr)
    adapter=ns['maybe_install_target_argmax_capture'](worker,enabled=True)
    ok('adapter registers before runner exists',adapter is not None and mr.capture_tail_hooks==[adapter])
    runner=type('DecodeCudaGraphRunner',(),{})();runner.model_runner=mr;runner.device_module=torch.cuda
    runner.capture_forward_mode=mode;runner.captured_req_width=8;runner.max_bs=10;runner.bs=4;runner.raw_bs=4;runner.raw_num_token=32
    for name in ['require_gathered_buffer','require_mlp_sync','enable_two_batch_overlap','enable_prefill_cp','enable_pdmux','record_nolora_graph','ragged_verify_mode','is_dllm']:
        setattr(runner,name,False)
    runner.attention_graph_variants=None;runner.model_context=contextlib.nullcontext
    runner.backend=backend_ns['FullCudaGraphBackend'](runner)
    mr.decode_cuda_graph_runner=runner
    # Exact capture wrapper from the inert overlay, including exception invalidation.
    capture_ns={};extract(OVER/'model_executor/runner/decode_cuda_graph_runner.py',capture_ns,['capture'],'DecodeCudaGraphRunner')
    runner.capture=lambda:capture_ns['capture'](runner)
    fb=NS(forward_mode=mode);capture_stream=torch.cuda.Stream();failure=False
    def capture_body():
        with runner.backend.capture_session(capture_stream):
            for rows in (80,32,16,8):
                key=Key(rows//8)
                def forward(rows=rows):
                    result=NS(next_token_logits=group.logits[0,:rows])
                    for hook in mr.capture_tail_hooks:hook(runner,result,fb,rows)
                    return result
                runner.backend.capture_one(key,forward)
                # This call is inserted directly after capture_one in the overlay.
                adapter.on_capture_success(runner,fb,rows,key)
                if failure:raise RuntimeError('injected after first graph installation')
    runner._capture=capture_body
    runner.capture()
    ok('all four actual backend graphs confirmed',len(adapter.confirmed_graphs)==4)
    def replay(rows):
        runner.bs=rows//8;runner.raw_bs=rows//8;runner.raw_num_token=rows;runner._replay_graph_key=Key(rows//8)
        runner.backend.replay(runner._replay_graph_key,fb)
        return adapter.consume(raw_rows=rows,graph_used=True,can_skip_verify_logits=True)
    for generation in range(4):
        group.logits.uniform_(-40,40)
        if generation==1:group.logits.zero_();group.logits[:,:,19]=2
        if generation==2:group.logits.fill_(-100);group.logits[:,:,3]=10000;group.logits[:,:,31]=10001
        for rows in (8,80,32,16,8):
            reference=lp.deferred_argmax(group.logits[0,:rows])
            ids=replay(rows)
            ok('native cap20 changed replay %d rows%d event%d'%(generation,rows,len(checks)),ids is not None and torch.equal(ids,reference))
    # Native default bucket transition B9 -> B10; only leading72tokenrows valid.
    for poison in ['nan_padding','different_padding_ids']:
        group.logits.fill_(-20)
        for row in range(72):group.logits[5,row,row+17]=20
        if poison=='nan_padding':group.logits[:,72:].fill_(float('nan'))
        else:group.logits[:,72:].fill_(10000)
        reference=lp.deferred_argmax(group.logits[0,:72])
        runner.bs=10;runner.raw_bs=9;runner.raw_num_token=72;runner._replay_graph_key=Key(10)
        runner.backend.replay(Key(10),fb)
        ids=adapter.consume(raw_rows=72,graph_used=True,can_skip_verify_logits=True)
        ok('paddedB9to10 returns72tokenrows '+poison,ids is not None and ids.shape==(72,) and torch.equal(ids,reference))
        ok('last realrequest retained '+poison,ids[-1].item()==5*25256+71+17)
        runner.raw_bs=8
        ok('inconsistent request/token counts reject '+poison,adapter.consume(raw_rows=72,graph_used=True,can_skip_verify_logits=True) is None)
    # Return to an exact bucket for the original lifecycle-negative checks.
    group.logits.uniform_(-40,40);replay(8)
    state=adapter.state;saved_output=state.output
    # Graph substitution must not authorize stale side output.
    key=runner._replay_graph_key;old_graph=runner.backend._graphs[key]
    runner.backend._graphs[key]=object()
    ok('graph substitution rejects side output',adapter.consume(raw_rows=8,graph_used=True,can_skip_verify_logits=True) is None)
    runner.backend._graphs[key]=old_graph
    failure=True
    try:runner.capture()
    except RuntimeError as err:ok('injected capture error preserved','injected' in str(err))
    else:raise AssertionError('missing capture error')
    ok('partial recapture invalidates all side outputs',not adapter.confirmed_graphs and not state._captures)
    ok('failed capture retains graph output buffer',adapter.state.output is saved_output)
    failure=False;runner.capture()
    ok('successful recapture requalifies all keys',len(adapter.confirmed_graphs)==4)
    group.logits.fill_(-20);group.logits[5,:,17]=20
    ok('post recapture side output equals eager',torch.equal(replay(16),lp.deferred_argmax(group.logits[0,:16])))
    ok('non greedy consume rejects',adapter.consume(raw_rows=16,graph_used=True,can_skip_verify_logits=False) is None)
    runner.backend.cleanup()
    ok('backend cleanup removes consumption authority',adapter.consume(raw_rows=16,graph_used=True,can_skip_verify_logits=True) is None)
    torch.cuda.synchronize()
    peak=torch.cuda.max_memory_allocated();assert peak<128*2**20
    paths=[Path(__file__),softpath,SRT/'layers/logits_processor.py',backend_path,
           OVER/'speculative/dflash_target_argmax_capture.py',OVER/'speculative/dflash_target_argmax_adapter.py',
           OVER/'model_executor/runner/decode_cuda_graph_runner.py']
    report={'status':'PASS_REAL_CUDA_PADDED_OVERLAY_LIFECYCLE_ONLY','checks':checks,'peak_allocated_bytes':peak,
            'elapsed_s':time.monotonic()-start,'padded_transition':'rawBS9 -> capturedBS10,72validtokenrows','target_softcap':20,'native_local_vocab':25256,
            'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'limitations':['TP8 locally emulated; no NCCL/NVLink','No model/full runner, memory-saver, graph pool borrowing or allocator integration',
                           'Actual backend methods with memory-pool scope and rank barrier stubs','No component timing or serving gain claim']}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['checks','source_hashes']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
