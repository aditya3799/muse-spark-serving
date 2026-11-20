from pathlib import Path
p=Path(__file__).with_name("gpu_regions.py");s=p.read_text(encoding="utf-8-sig")
s=s.replace('B=1','EARLY=os.environ.get("OKC_EARLY_TAPS")=="1"\nif EARLY:\n    assert SMALL, "full producer replay on this GPU is limited to small chunks"\n    LABEL="early_smoke"\nB=1',1)
s=s.replace('    device=torch.device("cuda")','    import faulthandler\n    faulthandler.dump_traceback_later(45,repeat=True)\n    device=torch.device("cuda")',1)
s=s.replace('    fc=torch.nn.Linear','    print("allocating native FC",seed,flush=True)\n    fc=torch.nn.Linear',1)
s=s.replace('    fused=data(', '    print("building fused KV",flush=True)\n    fused=data(',1)
s=s.replace('    second=(HERE/"applied_regions"/WORK).read_text()', '    second=(HERE/("applied_early" if EARLY else "applied_regions")/WORK).read_text()')
s=s.replace('    active={}\n    basis=rnd((1024,40960)); inp=torch.empty((max(CHUNK,MIX_N+16),40960),dtype=bf,device=device)','''    active={}
    basis=torch.empty((1024,40960),device=device,dtype=torch.float32 if EARLY else bf).normal_(generator=gen)
    inp=None if EARLY else torch.empty((max(CHUNK,MIX_N+16),40960),dtype=bf,device=device)
    if EARLY:
        from target_entry import CAP,target as native_target
        ctx["CaptureHiddenMode"]=CAP
    print("allocating worker pools",flush=True)''')
s=s.replace('        workers[arm]=worker,pool','''        if EARLY:
            state=dict(packs=0,gathers=[],monitor=False)
            state["make"]=lambda fb:basis.index_select(0,torch.remainder(active["features"],1024)).view(-1,5,8192).permute(1,0,2)
            call=native_target(arm,state)
            def target_source(batch,call=call,state=state,**kwargs):
                state["ids"]=active["positions"];state["loc"]=active["locations"]
                return call(batch,**kwargs)
            worker.target_worker=S(forward_batch_generation=target_source)
            worker._can_omit_target_aux=True
        workers[arm]=worker,pool''')
s=s.replace('            result=worker.forward_batch_generation(batch)','''            if EARLY:batch.return_hidden_states_mode=CAP.NULL
            batch.hicache_consumer_index=0
            result=worker.forward_batch_generation(batch)''')
s=s.replace('candidate="window_prefix_region_repair"', 'candidate="early_target_window_region_repair" if EARLY else "window_prefix_region_repair"')
s=s.replace('(HERE/"region_candidate.diff").read_bytes()', '(HERE/("early_candidate.diff" if EARLY else "region_candidate.diff")).read_bytes()')
s=s.replace('    detail=dict(seed=seed,','    if EARLY:scope+=" Real TpModelWorker.forward_batch_generation, Muse target core and outer CausalLM, five FP32 residual fixtures and source packing; no full target transformer."\n    detail=dict(seed=seed,')
s=s.replace('        for scene in SCENES:\n            d=', '        for scene in SCENES:\n            d=')
s=s.replace('    return reform,run,snapshot,lambda:basis.normal_(generator=gen)','    return reform,run,snapshot,lambda:basis.normal_(generator=gen)')
p.write_text(s,encoding="utf8")

