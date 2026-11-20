"""Execute exact exported prefix AST with a pure-Python integer/Triton model.

This validates values, masks, ownership and address geometry, not GPU codegen.
The scan model uses a parallel doubling tree, unlike the original serial loop.
"""
import ast
import copy
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent.parent
REL = "python/sglang/kernels/ops/attention/metadata.py"
OLD = ROOT / "sglang" / REL
NEW = OUT / "candidate" / REL


def wrap(value, bits):
    return ((int(value) + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1))


class Scalar(int):
    def __new__(cls, value, bits=32):
        obj = int.__new__(cls, wrap(value, bits))
        obj.bits = bits
        return obj

    def to(self, bits):
        return Scalar(self, bits)

    def __add__(self, other):
        return Scalar(int(self) + int(other), self.bits)

    __radd__ = __add__

    def __sub__(self, other):
        return Scalar(int(self) - int(other), self.bits)


class Vec(list):
    def op(self, other, fn):
        values = other if isinstance(other, Vec) else [other] * len(self)
        return Vec(fn(a, b) for a, b in zip(self, values))

    def __add__(self, other):
        return self.op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.op(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self.op(other, lambda a, b: int(a) * int(b))

    def __lt__(self, other):
        return self.op(other, lambda a, b: a < b)

    def to(self, bits):
        return Vec(Scalar(x, bits) for x in self)


class Pointer:
    def __init__(self, data, offsets=0, bits=32, reads=None):
        self.data, self.offsets, self.bits = data, offsets, bits
        self.reads = reads if reads is not None else []

    def __add__(self, offsets):
        if isinstance(offsets, Vec):
            offsets = Vec(int(self.offsets) + int(x) for x in offsets)
        else:
            offsets = int(self.offsets) + int(offsets)
        return Pointer(self.data, offsets, self.bits, self.reads)


class TL:
    int32 = 32

    def __init__(self, pid):
        self.pid = pid

    def program_id(self, axis):
        return self.pid[axis]

    def arange(self, lo, hi):
        return Vec(Scalar(x) for x in range(lo, hi))

    def load(self, ptr, mask=True, other=0):
        if isinstance(ptr.offsets, Vec):
            masks = mask if isinstance(mask, Vec) else [mask] * len(ptr.offsets)
            return Vec(self.load(Pointer(ptr.data, x, ptr.bits, ptr.reads), m, other)
                       for x, m in zip(ptr.offsets, masks))
        if not mask:
            return Scalar(other, ptr.bits)
        assert 0 <= ptr.offsets < len(ptr.data), "out-of-bounds load"
        ptr.reads.append(ptr.offsets)
        return Scalar(ptr.data[ptr.offsets], ptr.bits)

    def store(self, ptr, value, mask=True):
        if isinstance(ptr.offsets, Vec):
            masks = mask if isinstance(mask, Vec) else [mask] * len(ptr.offsets)
            values = value if isinstance(value, Vec) else [value] * len(ptr.offsets)
            for offset, item, valid in zip(ptr.offsets, values, masks):
                self.store(Pointer(ptr.data, offset, ptr.bits), item, valid)
        elif mask:
            assert 0 <= ptr.offsets < len(ptr.data), "out-of-bounds store"
            ptr.data[ptr.offsets] = wrap(value, ptr.bits)

    def where(self, mask, yes, no):
        return Vec(y if m else Scalar(no) for m, y in zip(mask, yes))

    def cumsum(self, values, axis):
        assert axis == 0
        result = Vec(values)
        distance = 1
        while distance < len(result):
            previous = Vec(result)
            for i in range(distance, len(result)):
                result[i] = previous[i] + previous[i - distance]
            distance *= 2
        return result

    def sum(self, values, axis):
        assert axis == 0
        values = list(values)
        while len(values) > 1:
            values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
        return values[0]


def dump(node):
    return ast.dump(node, include_attributes=False)


def read_functions(path):
    module = ast.parse(path.read_text(encoding="utf-8"))
    return module, {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}


def make_prefix_code(fn):
    # This exact source prefix contains only IDs and the owner-block if.
    assert isinstance(fn.body[2], ast.If)
    return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(fn.body[:3]),
                   type_ignores=[])), "exact_prefix", "exec")


def static_checks(old_module, new_module, old_fns, new_fns):
    changed = {name for name in old_fns if dump(old_fns[name]) != dump(new_fns[name])}
    assert changed == {"_fused_decode_metadata_kernel", "normal_decode_set_metadata"}
    old = old_fns["_fused_decode_metadata_kernel"]
    new = new_fns["_fused_decode_metadata_kernel"]
    assert [dump(n) for n in old.body[3:]] == [dump(n) for n in new.body[3:]]
    scan_if = new.body[2].body[0]
    assert [dump(n) for n in scan_if.orelse] == [dump(n) for n in old.body[2].body]
    restored_kernel = copy.deepcopy(new)
    restored_kernel.args.args.pop()
    restored_kernel.body[2].body = copy.deepcopy(old.body[2].body)
    assert dump(restored_kernel) == dump(old)
    restored_wrapper = copy.deepcopy(new_fns["normal_decode_set_metadata"])
    launch = next(n for n in ast.walk(restored_wrapper) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Subscript)
                  and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "_fused_decode_metadata_kernel")
    expression = next(k.value for k in launch.keywords if k.arg == "PREFIX_BLOCK")
    launch.keywords = [k for k in launch.keywords if k.arg != "PREFIX_BLOCK"]
    assert dump(restored_wrapper) == dump(old_fns["normal_decode_set_metadata"])
    # Prove all remaining top-level statements/imports are unchanged too.
    restored_module = copy.deepcopy(new_module)
    for i, node in enumerate(restored_module.body):
        if isinstance(node, ast.FunctionDef) and node.name in changed:
            restored_module.body[i] = copy.deepcopy(old_fns[node.name])
    assert dump(restored_module) == dump(old_module)
    assert max(map(len, NEW.read_text(encoding="utf-8").splitlines())) <= 400
    return compile(ast.Expression(body=expression), "exact_dispatch", "eval")


def choose_block(expression, batch, cache_stride, cu_stride):
    return eval(expression, dict(batch_size=batch,
        cache_seqlens_int32=SimpleNamespace(stride=lambda _: cache_stride),
        cu_seqlens_k=SimpleNamespace(stride=lambda _: cu_stride),
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length())))


def run_case(code, lengths, delta, strides, bits, block, pid=(0, 0), buffers=None):
    batch = len(lengths)
    seq_stride, cache_stride, cu_stride = strides
    sentinel = -123456789
    seq = [2 ** 45 + 17] * (max(1, batch) * max(1, seq_stride) + 8)
    for i, length in enumerate(lengths):
        seq[2 + i * seq_stride] = length
    if buffers is None:
        buffers = ([sentinel] * (max(1, batch) * max(1, cache_stride) + 8),
                   [sentinel] * ((batch + 1) * max(1, cu_stride) + 8))
    reads = []
    namespace = dict(tl=TL(pid), B=batch, PREFIX_BLOCK=block, SEQ_LEN_DELTA=delta,
        seq_lens=Pointer(seq, 2, bits, reads), seq_lens_stride_0=seq_stride,
        cache_seqlens_int32=Pointer(buffers[0], 2), cache_seqlens_int32_stride_0=cache_stride,
        cu_seqlens_k=Pointer(buffers[1], 2), cu_seqlens_k_stride_0=cu_stride)
    exec(code, namespace)
    assert set(reads).issubset({2 + i * seq_stride for i in range(batch)})
    return buffers


def main():
    old_module, old_fns = read_functions(OLD)
    new_module, new_fns = read_functions(NEW)
    dispatch = static_checks(old_module, new_module, old_fns, new_fns)
    old_code = make_prefix_code(old_fns["_fused_decode_metadata_kernel"])
    new_code = make_prefix_code(new_fns["_fused_decode_metadata_kernel"])
    rng = random.Random(576421)
    batches = [0, 1, 2, 3, 7, 8, 17, 31, 32, 65, 255, 513, 1023, 1024, 1025, 2049, 65537]
    checked = 0
    for batch in batches:
        for delta in [0, 1, 8, -9, 2 ** 31 - 1]:
            for strides in [(1, 1, 1), (3, 2, 4), (0, 1, 2), (1, 0, 1), (1, 1, 0)]:
                for bits in [32, 64]:
                    edge = [0, 1, 131072, 2 ** 31 - 1, 2 ** 31, -2 ** 31, -1,
                            2 ** 32 + 3, 2 ** 63 - 1, -2 ** 63]
                    lengths = [edge[i % len(edge)] for i in range(batch)]
                    block = choose_block(dispatch, batch, strides[1], strides[2])
                    assert (block == 0) == (batch > 1024 or not strides[1] or not strides[2])
                    old = run_case(old_code, lengths, delta, strides, bits, block)
                    new = run_case(new_code, lengths, delta, strides, bits, block)
                    assert old == new, (batch, delta, strides, bits)
                    checked += 1
    # Graph-like replay: reuse output storage and change every live input,
    # including valid padding rows with zero prefix (delta still applies).
    replay_checks = 0
    for batch in [1, 3, 17, 32, 65, 1024, 1025]:
        old_buffers = new_buffers = None
        block = choose_block(dispatch, batch, 2, 3)
        for step in range(9):
            lengths = [rng.randrange(0, 1048576) if i < batch - step % batch else 0
                       for i in range(batch)]
            old_buffers = run_case(old_code, lengths, 8, (2, 2, 3), 64, block, buffers=old_buffers)
            new_buffers = run_case(new_code, lengths, 8, (2, 2, 3), 64, block, buffers=new_buffers)
            assert old_buffers == new_buffers
            replay_checks += 1
        for pid in [(1, 0), (0, 1), (3, 4)]:
            before = copy.deepcopy(new_buffers)
            run_case(new_code, lengths, 8, (2, 2, 3), 64, block, pid=pid, buffers=new_buffers)
            assert new_buffers == before
    result = dict(passed=True, candidate_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),
        native_python_cases=checked, mutable_buffer_cases=replay_checks,
        non_owner_checks=21, batch_sizes=batches,
        arithmetic="signed int32 modular conversion/add/sub, int32/int64 source loads; parallel doubling scan vs original serial AST",
        static_checks="only prefix body/signature and one launch keyword differ; page logic/other functions unchanged; exact fallback AST",
        limitations="Pure-Python Triton semantic model only: not GPU codegen, actual Triton type lowering, CUDA graph replay or timing")
    (OUT / "cpu_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
