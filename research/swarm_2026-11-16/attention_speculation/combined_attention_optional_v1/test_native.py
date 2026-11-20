"""Native-only exact-method guard tests plus imported prefix AST regression."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
FA = "python/sglang/srt/layers/attention/flashattention_backend.py"
SOURCE = OUT / "candidate" / FA


def backend_class():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashAttentionBackend")
    selected = {"glue_draft_metadata", "_compute_scheduler_metadata", "_compute_verify_scheduler_metadata", "_set_decode_page_metadata"}
    class_copy = copy.deepcopy(cls)
    class_copy.bases = []
    class_copy.body = [n for n in class_copy.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    state = NS(algorithm="DFLASH", disable=False, cp=1)
    metadata_calls = []
    scope = dict(SpeculativeAlgorithm=NS(from_string=lambda s: NS(is_dflash=lambda: s == "DFLASH")),
        get_spec=lambda: NS(speculative_algorithm=state.algorithm),
        get_parallel=lambda: NS(attn_cp_size=state.cp),
        _should_disable_scheduler_metadata_precompute=lambda: state.disable,
        normal_decode_set_metadata=lambda *args, **kwargs: metadata_calls.append((args, kwargs)),
        IdSpaceKind=NS(SLIDING_WINDOW="swa"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), class_copy], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), scope)
    return scope["FlashAttentionBackend"], state, metadata_calls, cls


def fixture(cls):
    b = cls()
    vars(b).update(is_draft_runner=True, fa_impl_ver=4, topk=0,
        speculative_num_draft_tokens=8, is_encoder_decoder=False, use_mla=False,
        local_attn_builder=None, use_sliding_window_kv_pool=False,
        _kv_shard_pool=None, needs_cpu_seq_lens=False, is_prefill_aware_swa=False,
        kv_index_translator=NS(is_translating=False, reads_are_translated=False),
        _get_scheduler_metadata=None, _disable_scheduler_metadata_precompute=False,
        num_attention_heads=16, num_kv_heads=2, head_dim=64,
        kv_cache_dtype="bfloat16", page_size=256, has_softcap=False, num_splits=0)
    return b


def guard_tests():
    cls, state, metadata_calls, source_class = backend_class()
    b = fixture(cls)
    assert b.glue_draft_metadata() == ""
    glue_cases = 1
    for key, value in [("is_draft_runner", False), ("fa_impl_ver", 3), ("topk", 2),
            ("speculative_num_draft_tokens", None), ("is_encoder_decoder", True),
            ("use_mla", True), ("local_attn_builder", object()),
            ("use_sliding_window_kv_pool", True), ("_kv_shard_pool", object()),
            ("needs_cpu_seq_lens", True), ("is_prefill_aware_swa", True)]:
        old = getattr(b, key)
        setattr(b, key, value)
        assert b.glue_draft_metadata(), key
        setattr(b, key, old)
        glue_cases += 1
    for key in ["is_translating", "reads_are_translated"]:
        setattr(b.kv_index_translator, key, True)
        assert b.glue_draft_metadata(), key
        setattr(b.kv_index_translator, key, False)
        glue_cases += 1
    for key, bad in [("algorithm", "DSPARK"), ("disable", True), ("cp", 2)]:
        old = getattr(state, key)
        setattr(state, key, bad)
        assert b.glue_draft_metadata(), key
        setattr(state, key, old)
        glue_cases += 1
    b.has_swa, b.sliding_window_size = True, 2048
    assert b.glue_draft_metadata() == ""  # Ordinary MHA storage with SWA attention.
    glue_cases += 1

    scheduler_calls = []
    def scheduler(**kwargs):
        scheduler_calls.append(kwargs)
        return kwargs
    scheduler_cases = 0
    for batch in [1, 3, 4, 8, 32]:
        for page in [1, 16, 256]:
            for width in [1, 4, 8]:
                for dtype in ["bfloat16", "float8_e4m3fn"]:
                    for softcap in [False, True]:
                        b = fixture(cls)
                        b.fa_impl_ver = 3
                        b.is_draft_runner = False
                        b._get_scheduler_metadata = scheduler
                        b.page_size, b.kv_cache_dtype, b.has_softcap = page, dtype, softcap
                        metadata = NS(page_table=NS(shape=(batch, 73)), cache_seqlens_int32=object(),
                            cu_seqlens_q=object(), max_seq_len_q=width, max_seq_len_k=1)
                        result = b._compute_verify_scheduler_metadata(metadata, batch)
                        assert result["max_seqlen_k"] == 73 * page  # Capacity, not stale/live host max.
                        assert result["max_seqlen_q"] == width
                        assert result["qkv_dtype"] == dtype and result["has_softcap"] == softcap
                        assert result["causal"] is True and result["page_size"] == page
                        assert result["cache_seqlens"] is metadata.cache_seqlens_int32
                        assert result["cu_seqlens_q"] is metadata.cu_seqlens_q
                        assert result["batch_size"] == batch
                        assert (result["num_heads"], result["num_heads_k"], result["headdim"]) == (16, 2, 64)
                        assert b.glue_draft_metadata()  # FA3 target cannot enter FA4 draft glue.
                        scheduler_cases += 1
    for attr, value in [("local_attn_builder", object()), ("_kv_shard_pool", object()),
            ("topk", 2), ("use_mla", True), ("_disable_scheduler_metadata_precompute", True),
            ("_get_scheduler_metadata", None)]:
        old = getattr(b, attr)
        count = len(scheduler_calls)
        setattr(b, attr, value)
        assert b._compute_verify_scheduler_metadata(metadata, 32) is None
        assert len(scheduler_calls) == count
        setattr(b, attr, old)
        scheduler_cases += 1
    b = fixture(cls)  # Actual FA4 construction has no FA3 scheduler callable.
    assert b.glue_draft_metadata() == ""
    assert b._compute_verify_scheduler_metadata(metadata, 32) is None
    scheduler_cases += 1

    forward = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend")
    consume = next(n for n in ast.walk(forward) if isinstance(n, ast.If) and len(n.body) == 1
        and isinstance(n.body[0], ast.Assign) and isinstance(n.body[0].value, ast.Attribute)
        and n.body[0].value.attr == "scheduler_metadata")
    guard = compile(ast.Expression(body=consume.test), "exact_scheduler_consumer_guard", "eval")
    b = fixture(cls)
    environment = dict(self=b, forward_batch=NS(forward_mode=NS(is_target_verify=lambda: True)),
        metadata=NS(scheduler_metadata=object(), image_token_ranges=None), causal=True,
        is_swa_layer=False, use_local_attn=False, layer=NS(is_cross_attention=False,
            tp_q_head_num=16, tp_k_head_num=2, head_dim=64, v_head_dim=64, logit_cap=0),
        sinks=None, score_mod=None, rel_bias=None, q=NS(dtype="bfloat16"))
    assert eval(guard, environment)
    consumer_cases = 1
    for name, bad in [("causal", False), ("is_swa_layer", True), ("use_local_attn", True),
            ("sinks", object()), ("score_mod", object()), ("rel_bias", object())]:
        old = environment[name]
        environment[name] = bad
        assert not eval(guard, environment), name
        environment[name] = old
        consumer_cases += 1
    for name, attr, bad in [("self", "topk", 2), ("metadata", "scheduler_metadata", None),
            ("metadata", "image_token_ranges", object()), ("q", "dtype", "float16"),
            ("layer", "is_cross_attention", True), ("layer", "tp_q_head_num", 8),
            ("layer", "tp_k_head_num", 1), ("layer", "head_dim", 128),
            ("layer", "v_head_dim", 128), ("layer", "logit_cap", 10)]:
        obj = environment[name]
        old = getattr(obj, attr)
        setattr(obj, attr, bad)
        assert not eval(guard, environment), (name, attr)
        setattr(obj, attr, old)
        consumer_cases += 1
    environment["forward_batch"].forward_mode.is_target_verify = lambda: False
    assert not eval(guard, environment)
    consumer_cases += 1

    # Exercise the exact metadata forwarding method; GPU kernels are a spy here.
    b = fixture(cls)
    b.req_to_token, b.max_num_pages = object(), 73
    m = NS(cache_seqlens_int32=object(), cu_seqlens_k=object(), page_table=object(), swa_page_table=None)
    req, lens, plan = object(), object(), object()
    b._set_decode_page_metadata(plan, m, req, lens, 8)
    args, kwargs = metadata_calls[-1]
    assert args == (m.cache_seqlens_int32, m.cu_seqlens_k, m.page_table,
        b.req_to_token, req, 73, lens, 8, 256, None)
    assert kwargs == dict(full_to_swa_index_mapping=None, skip_page_table=False)
    copies = []
    b.kv_index_translator.reads_are_translated = True
    b.kv_index_translator.copy_page_table = lambda *args, **kw: copies.append((args, kw))
    req = NS(numel=lambda: 3)
    class Table:
        def __getitem__(self, key): return (self, key.stop)
    m.page_table, m.swa_page_table = Table(), Table()
    b._set_decode_page_metadata(plan, m, req, lens, 8)
    assert metadata_calls[-1][1]["skip_page_table"] is True
    assert len(copies) == 2 and copies[1][1]["kind"] == "swa"
    assert b.glue_draft_metadata()
    return dict(passed=True, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        glue_guard_cases=glue_cases, scheduler_geometry_and_guard_cases=scheduler_cases,
        consumer_guard_cases=consumer_cases, exact_metadata_forwarding_cases=2,
        limitations="Exact extracted methods and guard AST with native spies, not tensor kernels or graph execution")


def main():
    report = guard_tests()
    (OUT / "guard_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    spec = importlib.util.spec_from_file_location("combined_prefix_checks", BASE / "prefix_scan/test_cpu.py")
    prefix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prefix)
    prefix.OUT = OUT
    prefix.NEW = OUT / "candidate/python/sglang/kernels/ops/attention/metadata.py"
    prefix.main()
    results = dict(guards=report, prefix=json.loads((OUT / "cpu_validation.json").read_text(encoding="utf-8")))
    (OUT / "combined_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
