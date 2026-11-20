"""Bounded correctness only. Requires explicit exclusive GPU handoff."""
import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parent.parent
FA = OUT / "candidate/python/sglang/srt/layers/attention/flashattention_backend.py"
META = OUT / "candidate/python/sglang/kernels/ops/attention/metadata.py"
EXPECTED = {FA: "0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e",
            META: "300cb9428d6281c10812be0a302e88d82dff9f81baf5cab1087730fe8bc6df3c"}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wrap32(value):
    return ((int(value) + 2 ** 31) % 2 ** 32) - 2 ** 31


def prefix_cases(torch, old, new, bound):
    checked = 0
    for batch in [0, 1, 3, 8, 17, 32, 65, 1024, 1025, 2049]:
        for dtype in [torch.int32, torch.int64]:
            bound()
            extremes = [0, 1, 131072, 2 ** 31 - 1, -2 ** 31, -1]
            if dtype == torch.int64:
                extremes += [2 ** 32 + 7, 2 ** 63 - 1, -2 ** 63]
            source = torch.tensor([extremes[i % len(extremes)] for i in range(batch)], dtype=dtype)
            backing = torch.zeros(max(1, batch) * 3 + 8, dtype=dtype, device="cuda")
            seq = backing[2:2 + batch * 3:3]
            seq.copy_(source)
            dummy = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
            req = torch.zeros(batch, dtype=torch.int32, device="cuda")
            for delta in [0, 8, -9, 2 ** 31 - 1]:
                bound()
                outputs = []
                for fn in [old.normal_decode_set_metadata, new.normal_decode_set_metadata]:
                    cache_storage = torch.full((max(1, batch) * 2 + 8,), -9876543, dtype=torch.int32, device="cuda")
                    cu_storage = torch.full(((batch + 1) * 3 + 8,), -9876543, dtype=torch.int32, device="cuda")
                    cache = cache_storage[2:2 + batch * 2:2]
                    cu = cu_storage[2:2 + (batch + 1) * 3:3]
                    fn(cache, cu, dummy, dummy, req, 0, seq, delta, 1, skip_page_table=True)
                    outputs.append((cache_storage, cu_storage, cache, cu))
                torch.cuda.synchronize()
                assert torch.equal(outputs[0][0], outputs[1][0]), (batch, dtype, delta, "cache")
                assert torch.equal(outputs[0][1], outputs[1][1]), (batch, dtype, delta, "prefix")
                expected = [wrap32(v + delta) for v in source.tolist()]
                sums = [0]
                for v in expected:
                    sums.append(wrap32(sums[-1] + v))
                assert outputs[1][2].cpu().tolist() == expected
                assert outputs[1][3].cpu().tolist() == sums
                checked += 1
    return checked


def graph_cases(torch, old, new, bound):
    cases = {}
    for page, mapping in [(1, "none"), (16, "static_swa"), (256, "translated")]:
        for batch in [1, 3, 8, 17]:
            bound()
            pages, width, slots = 32, 32 * page, batch + 4
            req_table = torch.arange(slots * width, device="cuda", dtype=torch.int32).view(slots, width)
            req = torch.arange(batch, device="cuda", dtype=torch.int32)
            seq = torch.full((batch,), min(width - 8, 17), device="cuda", dtype=torch.int32)
            static = torch.arange(slots * width, device="cuda", dtype=torch.int32).flip(0) if mapping == "static_swa" else None
            full_v2p = torch.arange(slots * pages, device="cuda", dtype=torch.int32).flip(0) if mapping == "translated" else None
            swa_v2p = full_v2p.clone() if full_v2p is not None else None
            args = dict(req_to_token=req_table, req_pool_indices=req, max_seq_pages=pages,
                        seq_lens=seq, seq_len_delta=8, page_size=page,
                        full_to_swa_index_mapping=static, full_v2p=full_v2p, swa_v2p=swa_v2p,
                        zero_tail=mapping == "translated")
            buffers = []
            for _ in range(2):
                buffers.append(dict(cache_seqlens_int32=torch.zeros(batch, device="cuda", dtype=torch.int32),
                    cu_seqlens_k=torch.zeros(batch + 1, device="cuda", dtype=torch.int32),
                    page_table=torch.full((batch, pages), -91, device="cuda", dtype=torch.int32),
                    swa_page_table=torch.full((batch, pages), -91, device="cuda", dtype=torch.int32) if mapping != "none" else None))
            for _ in range(3):
                old.normal_decode_set_metadata(**buffers[0], **args)
                new.normal_decode_set_metadata(**buffers[1], **args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                new.normal_decode_set_metadata(**buffers[1], **args)
            cases[page, batch] = (args, buffers, graph)
    checked = 0
    for page in [1, 16, 256]:
        for batch in [1, 3, 8, 17, 3, 1]:
            args, buffers, graph = cases[page, batch]
            for step in range(3):
                bound()
                args["seq_lens"].copy_((torch.arange(batch, device="cuda", dtype=torch.int32) * 79 + step * 107) % (32 * page - 8))
                args["req_pool_indices"].copy_((torch.arange(batch, device="cuda", dtype=torch.int32) + step + 1) % (batch + 4))
                old.normal_decode_set_metadata(**buffers[0], **args)
                graph.replay()
                torch.cuda.synchronize()
                for key in buffers[0]:
                    if buffers[0][key] is not None:
                        assert torch.equal(buffers[0][key], buffers[1][key]), (page, batch, step, key)
                checked += 1
    del cases, graph, buffers
    torch.cuda.synchronize()
    return checked


def combined_glue_case():
    # Reuse the held SWA fixture but point both backend and metadata module
    # at the combined exact exports, without changing frozen test files.
    path = BASE / "draft_glue/test_glue.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    overrides = {"OUT": OUT, "SGL": ROOT / "sglang/python/sglang", "SOURCE": FA}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in overrides:
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(overrides[name]))], keywords=[])
            elif name == "metadata_kernel":
                node.value.args[0] = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(META))], keywords=[])
    # Match the actual FlashAttentionMetadata dataclass default introduced
    # into this call path by combining the scheduler and glue components.
    defaults_added = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "m"
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "NS"):
            assert not any(k.arg == "scheduler_metadata" for k in node.value.keywords)
            node.value.keywords.append(ast.keyword(arg="scheduler_metadata", value=ast.Constant(None)))
            defaults_added += 1
    assert defaults_added == 1
    scope = {"__name__": "combined_held_glue_fixture", "__file__": str(path)}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), scope)
    assert scope["SOURCE"] == FA
    assert Path(scope["metadata_kernel"].__file__) == META
    return scope["run"]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-authorized", action="store_true")
    args = parser.parse_args()
    if not args.gpu_authorized:
        raise RuntimeError("Exclusive GPU handoff required")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in EXPECTED}
    assert all(hashes[str(p)] == h for p, h in EXPECTED.items())
    import torch
    started = time.monotonic()
    def bound():
        if time.monotonic() - started > 210:
            raise TimeoutError("Correctness budget exceeded")
    report = dict(passed=False, hashes=hashes, device=torch.cuda.get_device_name(),
        scope="Exact combined prefix GPU output and replay; explicit-SWA glue metadata; no timing/model acceptance")
    path = OUT / "gpu_validation.json"
    def save():
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        old = load(ROOT / "sglang/python/sglang/kernels/ops/attention/metadata.py", "original_prefix_metadata")
        new = load(META, "combined_prefix_metadata")
        report["prefix_cases"] = prefix_cases(torch, old, new, bound)
        save(); print(json.dumps({"prefix_cases": report["prefix_cases"]}), flush=True)
        report["graph_cases"] = graph_cases(torch, old, new, bound)
        save(); print(json.dumps({"graph_cases": report["graph_cases"]}), flush=True)
        bound()
        report["glue"] = combined_glue_case()
        torch.cuda.synchronize()
        report["passed"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
