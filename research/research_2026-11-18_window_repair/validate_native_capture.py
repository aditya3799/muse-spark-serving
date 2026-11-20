from pathlib import Path
b=Path(__file__).resolve().parent
p=b/"build_early.py";s=p.read_text(encoding="utf8")
s=s.replace('not any(getattr(m, "expected_aux_hidden_states", None) is not None for m in target.model.modules())','not any(getattr(m, "keep_last_hidden_states", False) or getattr(m, "expected_aux_hidden_states", None) is not None\n                for m in target.model.modules())')
p.write_text(s,encoding="utf8")
p=b/"target_entry.py";s=p.read_text(encoding="utf8")
s=s.replace('    def need_capture(self):return self.flag','    def need_capture(self):return self.flag\n    def is_full(self):return self.flag')
s=s.replace('        state["packs"]+=1','        if not isinstance(aux,torch.Tensor):state["packs"]+=1')
s=s.replace('    def logits(ids,hidden,head,fb,aux):','''    choose=extract(stock("python/sglang/srt/layers/logits_processor.py"),"LogitsProcessor","_get_hidden_states_to_store",dict(torch=torch,pack_aux_hidden_states=tracked_pack))
    def logits(ids,hidden,head,fb,aux):''')
s=s.replace('z=tracked_pack(aux) if aux is not None and fb.capture_hidden_mode.need_capture() and not isinstance(aux,torch.Tensor) else aux','z=choose(None,hidden,None,aux,hidden,None,None,selections,fb)')
s=s.replace('b.logits_output.hidden_states is None and not y["gathers"]','b.logits_output.hidden_states.shape==base.shape[1:] and not y["gathers"]')
s=s.replace('result.logits_output.hidden_states is None and state["packs"]','result.logits_output.hidden_states.shape==base.shape[1:] and state["packs"]')
p.write_text(s,encoding="utf8")

