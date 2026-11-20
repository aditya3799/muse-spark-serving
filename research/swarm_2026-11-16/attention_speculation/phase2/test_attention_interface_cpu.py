"""Native CPU call-boundary proof; explicitly no attention arithmetic or CUDA."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

OUT = Path(__file__).resolve().parent
SWARM = OUT.parent.parent
SCORED = SWARM / 'coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/layers/attention/flashattention_backend.py'
OPTION = OUT.parent / 'combined_attention_optional_v1/candidate/python/sglang/srt/layers/attention/flashattention_backend.py'
PINS = {SCORED: '2a720c8f5ed86437e6ec181cace03ae97fba0460672b1d001549f8df4b71e03c',
        OPTION: '0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e'}


class TensorToken:
    """Tracks view identity only; does not pretend to implement tensor math."""
    def __init__(self, label):
        self.label = label
    def contiguous(self):
        return self
    def view(self, *shape):
        return self


def method_map(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlashAttentionBackend')
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def run():
    for path, digest in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    old, new = method_map(SCORED), method_map(OPTION)
    # This optional FA source changes eligibility only relative to scored FA.
    assert set(new) - set(old) == {'glue_draft_metadata'}
    assert not (set(old) - set(new))
    for name in old:
        assert ast.dump(old[name]) == ast.dump(new[name]), name
    calls, writes = [], []
    def attention(**kwargs):
        calls.append(kwargs)
        return TensorToken('attention-output')
    scope = dict(is_cp_active=lambda fb: False,
                 AttentionType=NS(ENCODER_ONLY='encoder', DECODER_BIDIRECTIONAL='bidirectional'),
                 KVWriteLoc=NS(for_layer=lambda fb, layer, swa_loc: fb.out_cache_loc),
                 flash_attn_with_kvcache=attention)
    method = old['forward_extend']
    cls = ast.ClassDef(name='Backend', bases=[], keywords=[], body=[method], decorator_list=[])
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(SCORED), 'exec'), scope)
    b = scope['Backend']()
    b.fa_impl_ver = 4
    b.use_mla = False
    b.fa_skip_kv_cache = False
    b.kv_cache_is_mxfp8 = False
    b.use_sliding_window_kv_pool = False
    b.local_attn_builder = None
    b.topk = 0
    b.num_splits = 0
    b.full_cg_prefill_metadata = None
    b.prepare_paged_mha_query = lambda q, qr, kr, layer, **kw: (q, qr, kr, None, None)
    b._mxfp8_sf_kwargs = lambda *args: {}
    kpool, vpool = TensorToken('BF16-K-pool'), TensorToken('BF16-V-pool')
    b.get_paged_mha_kv_cache = lambda layer: (kpool, vpool)
    b.token_to_kv_pool = NS(set_kv_buffer=lambda *args: writes.append(args))
    cases = 0
    for bs in (1, 4, 8):
        for window in (-1, 2047):
            for kind in ('decoder', 'bidirectional'):
                for save in (False, True):
                    for iteration in (0, 1):
                        # New objects make accidental stale metadata observable at the boundary.
                        m = NS(image_token_ranges=None, swa_spec_metadata=None, swa_out_cache_loc=None, scheduler_metadata=None,
                               page_table=TensorToken('page-%s' % iteration),
                               cache_seqlens_int32=TensorToken('length-%s' % iteration),
                               cu_seqlens_q=TensorToken('q-cu-%s' % iteration),
                               cu_seqlens_k=TensorToken('k-cu-%s' % iteration), max_seq_len_q=8)
                        b.forward_metadata = m
                        loc = TensorToken('write-%s' % iteration)
                        fb = NS(batch_size=bs, forward_mode=NS(is_target_verify=lambda: True), out_cache_loc=loc)
                        layer = NS(sliding_window_size=window, is_cross_attention=False, attn_type=kind,
                                   tp_q_head_num=4, tp_k_head_num=1, head_dim=128, v_head_dim=128,
                                   scaling=128 ** -0.5, logit_cap=0.0, k_scale=None, v_scale=None)
                        q, k, v = [TensorToken(x) for x in ('q', 'k', 'v')]
                        before = len(writes)
                        output = b.forward_extend(q, k, v, layer, fb, save_kv_cache=save)
                        a = calls[-1]
                        assert output.label == 'attention-output'
                        for key, expected in dict(q=q, k_cache=kpool, v_cache=vpool,
                                                  page_table=m.page_table, cache_seqlens=m.cache_seqlens_int32,
                                                  cu_seqlens_q=m.cu_seqlens_q, cu_seqlens_k_new=m.cu_seqlens_k).items():
                            assert a[key] is expected, key
                        causal = kind == 'decoder'
                        assert a['causal'] is causal
                        assert a['window_size'] == ((window, 0 if causal else window) if window >= 0 else (-1, -1))
                        assert a['ver'] == 4 and a['max_seqlen_q'] == 8
                        assert a['softmax_scale'] == layer.scaling and a['softcap'] == 0.0
                        assert a['num_splits'] == 0 and a['return_softmax_lse'] is False
                        assert 'k_descale' not in a and 'v_descale' not in a
                        assert len(writes) - before == int(save)
                        if save:
                            assert writes[-1] == (layer, loc, k, v, None, None)
                        cases += 1
    return dict(passed=True, cases=cases, unchanged_scored_methods=len(old),
                hashes={str(p): d for p, d in PINS.items()},
                scope='Exact unchanged forward_extend method boundary and KV-write routing with symbolic tensors; no tensor math, CUDA graph execution, FA4 arithmetic, dtype runtime check or generation acceptance proof')


if __name__ == '__main__':
    result = run()
    (OUT / 'attention_interface_cpu.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))
