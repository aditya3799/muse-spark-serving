from pathlib import Path
b=Path(__file__).resolve().parent
p=b/"build_early.py";s=p.read_text(encoding="utf-8-sig")
s=s.replace('modify(muse,[("        aux_hidden_states', 'modify(muse,[("class MuseSparkTextForCausalLM(nn.Module):\\n","class MuseSparkTextForCausalLM(nn.Module):\\n    supports_dflash_window_aux_omission = True\\n"),("        aux_hidden_states')
s=s.replace('        if self.capture_aux_hidden_states and not getattr(forward_batch, \\"dflash_skip_prefill_aux\\", False):\\n            hidden_states, aux_hidden_states = hidden_states','        if self.capture_aux_hidden_states:\\n            if getattr(forward_batch, \\"dflash_skip_prefill_aux\\", False):\\n                if isinstance(hidden_states, tuple):\\n                    # A replay of a statically captured body may still provide taps.\\n                    hidden_states = hidden_states[0]\\n            else:\\n                hidden_states, aux_hidden_states = hidden_states')
s=s.replace('        from sglang.srt.model_executor.runner import EagerRunner\n','')
s=s.replace('        prefill_graph = target.prefill_cuda_graph_runner\n','')
s=s.replace('(prefill_graph is None or isinstance(prefill_graph, EagerRunner))','any(getattr(m, "supports_dflash_window_aux_omission", False) for m in target.model.modules())')
s=s.replace('capture_hidden_mode=CaptureHiddenMode.NULL if skip_aux else CaptureHiddenMode.FULL,','capture_hidden_mode=CaptureHiddenMode.FULL,')
p.write_text(s,encoding="utf8")
p=b/"target_entry.py";s=p.read_text(encoding="utf-8-sig").replace('capture_hidden_mode=CAP.NULL if omit else CAP.FULL','capture_hidden_mode=CAP.FULL')
s=s.replace('return forward(inner,ids,pos,fb,input_embeds=embed)','''if state.get("captured_body"):
            fb=S(forward_mode=fb.forward_mode)
        return forward(inner,ids,pos,fb,input_embeds=embed)''')
s=s.replace('    (H/"early_target_cpu.json")','''    state=dict(make=lambda fb:base,ids=torch.arange(12),loc=torch.arange(12),packs=0,gathers=[],logits=[],monitor=True,captured_body=True)
    call=target("candidate",state,True)
    result=call(bch, capture_hidden_mode=CAP.FULL,return_kv_loc_plan=True,dflash_skip_prefill_aux=True)
    assert result.logits_output.hidden_states is None and state["packs"]==0 and state["gathers"]==[]
    assert torch.equal(result.next_token_ids,a.next_token_ids)
    d["captured_body_with_eager_outer"]={"original_five_taps_replayed":True,"outer_gathers":state["gathers"],"packs":state["packs"]}
    (H/"early_target_cpu.json")'''.replace("call(bch,","call(S(hicache_consumer_index=0,extend_lens=[7,5],seq_lens=torch.tensor([7,5]),forward_mode=None),"))
p.write_text(s,encoding="utf8")

