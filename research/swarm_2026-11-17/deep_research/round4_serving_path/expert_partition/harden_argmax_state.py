from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'dflash_target_argmax_adapter.py'; s=p.read_text(encoding='utf-8-sig')
s=s.replace('def _plain_key(key):', '''def _lp_key(lp):
    # Compare only scalar Python values, never tensors or overloaded equality.
    cap = lp.final_logit_softcapping
    vocab = lp.vocab_size
    deferred = lp.defer_verify_gather
    if ((cap is not None and type(cap) not in (int, float))
            or type(vocab) is not int or type(deferred) is not bool):
        return None
    if cap is not None and not (-float('inf') < cap < float('inf')):
        return None
    return (id(lp), cap, vocab, deferred)


def _plain_key(key):''')
s=s.replace('        self.topology = None\n', '        self.topology = None\n        self.lp_key = None\n        self.forward_stream_key = None\n')
s=s.replace('        rows = runner.max_bs * runner.captured_req_width', '''        lp_key = _lp_key(self.worker._target_logits_processor)
        if lp_key is None:
            self.disabled_reason = 'unsupported_logits_processor_scalars'
            return
        rows = runner.max_bs * runner.captured_req_width''')
s=s.replace('        self.topology = _topology_key()\n', '''        self.topology = _topology_key()
        self.lp_key = lp_key
        self.forward_stream_key = self.model_runner.forward_stream.cuda_stream
''')
s=s.replace('or _topology_key() != self.topology)', 'or _topology_key() != self.topology\n                or _lp_key(self.worker._target_logits_processor) != self.lp_key)')
s=s.replace('or _topology_key() != self.topology or not _plain_key(shape_key)', 'or _topology_key() != self.topology\n                or _lp_key(self.worker._target_logits_processor) != self.lp_key\n                or not _plain_key(shape_key)')
s=s.replace("        key = getattr(runner, '_replay_graph_key', None)", '''        # Capture uses a private stream; never compare with that stream.
        # Scored overlap execution calls target and this consumer inside the
        # target model runner's stable forward_stream context. Other streams
        # fall back, including unsupported non-overlap/default-stream use.
        if (self.model_runner.forward_stream.cuda_stream != self.forward_stream_key
                or torch.cuda.current_stream(self.worker.device).cuda_stream
                != self.forward_stream_key):
            return None
        key = getattr(runner, '_replay_graph_key', None)''')
p.write_text(s)
p=H/'test_target_argmax_overlay.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('cuda=NS(capturing=False);', 'cuda=NS(capturing=False,stream_key=101); cuda.current_stream=lambda device:NS(cuda_stream=cuda.stream_key);')
s=s.replace('lp=NS(defer_verify_gather=True,', 'lp=NS(final_logit_softcapping=30.0,vocab_size=256,defer_verify_gather=True,')
s=s.replace('mr=NS(is_draft_worker=False,', 'mr=NS(forward_stream=NS(cuda_stream=101),is_draft_worker=False,')
s=s.replace("old=r.backend._graphs[Key()]", '''for field,value in [('final_logit_softcapping',31.0),('vocab_size',255),('defer_verify_gather',False)]:
    lp=w._target_logits_processor; oldvalue=getattr(lp,field); setattr(lp,field,value); ok('mutated LP '+field,consume(a) is None); setattr(lp,field,oldvalue)
oldlp=w._target_logits_processor; w._target_logits_processor=NS(**vars(oldlp)); ok('replaced LP object rejected',consume(a) is None); w._target_logits_processor=oldlp
cuda.stream_key=202; ok('nonforward consumption stream rejected',consume(a) is None); cuda.stream_key=101
m.forward_stream=NS(cuda_stream=202); ok('replaced forward stream rejected',consume(a) is None); m.forward_stream=NS(cuda_stream=101)
old=r.backend._graphs[Key()]''')
p.write_text(s)
