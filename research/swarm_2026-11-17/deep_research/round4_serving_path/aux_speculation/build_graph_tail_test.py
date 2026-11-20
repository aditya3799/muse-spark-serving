from pathlib import Path
P=Path(__file__).parent
s=(P/'test_prefix_worker_tail.py').read_text()
s=s.replace("import dflash_prefix_verification as support","import dflash_prefix_verification as support\nfrom target_verify_runtime_state import TargetVerifyStates\nfrom target_prefix_graph_runtime import target_prefix_scope")
s=s.replace("HERE/'eager_prefix_overlay/python/sglang/srt'","HERE/'graph_prefix_overlay/python/sglang/srt'")
s=s.replace("'resolve_prefix_verification_width':support.resolve_prefix_verification_width,","'resolve_prefix_verification_width':support.resolve_prefix_verification_width,\n      'target_prefix_scope':lambda worker,width:target_prefix_scope(worker,width,stream_key=17),")
s=s.replace('assert torch.equal(backend.forward_metadata.cache_seqlens_int32,(prefix+w).to(torch.int32))','assert torch.equal(runner.attn_backend.forward_metadata.cache_seqlens_int32,(prefix+w).to(torch.int32))\n        if candidate: assert worker._prefix_target_states.active')
s=s.replace("can_run_cuda_graph=(w==8)","can_run_cuda_graph=True")
old="    worker._draft_write_ids=lambda p:p.write_ids(reader)"
new='''    if candidate:
        runner.init_new_workspace=False;runner.is_draft_worker=False
        for key in ('model','req_to_token_pool','token_to_kv_pool','token_to_kv_pool_allocator','kv_index_translator','prefill_cuda_graph_runner','graph_shared_output'):
            setattr(runner,key,object())
        manager=TargetVerifyStates(runner,8,None)
        def make_backend(mr):
            b=FlashAttentionBackend(8,table,reader,swa=True)
            b.target_verify_metadata={};b._verify_mask=None
            return b
        def make_graph(mr,b,w):
            g=scaffold.Graph(w,b);g.attn_backend=b;g.model_runner=mr
            return g
        if width!=8: manager.build(width,make_backend,make_graph)
        manager.seal_startup();worker._prefix_target_states=manager
    worker._draft_write_ids=lambda p:p.write_ids(reader)'''
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("        events.append('append')","        if candidate: assert worker._prefix_target_states.active\n        events.append('append')")
s=s.replace("        events.append('publish')","        if candidate: assert worker._prefix_target_states.active\n        events.append('publish')")
s=s.replace("    except ValueError as error:\n        if reject:","    except ValueError as error:\n        if candidate:\n            assert not manager.active and runner.attn_backend is backend\n        if reject:")
s=s.replace("    assert not reject, 'padded outputs were admitted'","    if candidate:\n        assert not manager.active and runner.attn_backend is backend\n    assert not reject, 'padded outputs were admitted'")
s=s.replace('CPU_SOURCE_EXTRACTED_WORKER_TAIL_PASS','CPU_GRAPH_SCOPED_SOURCE_WORKER_TAIL_PASS').replace("HERE/'eager_prefix_overlay.diff'","HERE/'graph_prefix_overlay.diff'").replace('PREFIX_WORKER_TAIL_RESULT.json','GRAPH_SCOPED_WORKER_TAIL_RESULT.json')
s=s.replace("'Graph load/device page-table builder'","'Fake Graph.load_batch invokes extracted actual FA3 metadata with CPU builder; not actual DecodeCudaGraphRunner/model'")
(P/'test_graph_scoped_worker_tail.py').write_text(s)