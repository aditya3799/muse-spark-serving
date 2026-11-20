"""CPU-only bit audit of public Humming PRs 103 and 107 on the scored NVFP4 path."""
import json
import math
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bits(value):
    try:
        word = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError:
        word = 0xff800000 if value < 0 else 0x7f800000
    if word & 0x7f800000 == 0x7f800000:
        return word >> 16
    return (word + 0x7fff + ((word >> 16) & 1)) >> 16


def value(code):
    return struct.unpack("<f", struct.pack("<I", code << 16))[0]


def bf(x):
    return value(bits(x))


def scale(code):
    sign = -1 if code >= 128 else 1
    exponent = (code & 127) >> 3
    mantissa = code & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    return sign * (math.ldexp(1 + mantissa / 8, exponent - 7)
                   if exponent else mantissa * 2.**-9)


def weight(code):
    return (-1 if code >= 8 else 1) * (0., .5, 1., 1.5, 2., 3., 4., 6.)[code & 7]


def output(a, operand, restore):
    return bits(bf(a * operand) * restore)


def main():
    checked = 0
    for q in range(16):
        for s in range(256):
            v = scale(s)
            if not math.isfinite(v):
                continue
            raw_q = bf(weight(q) * 2.**-126)
            raw_s = bf(v * 2.**-120)
            before = bf(bf(raw_q * 2.**124) * bf(raw_s * 2.**120))
            folded_s = bf(bf(raw_s * 2.**127) * 2.**112)
            after = bf(raw_q * folded_s)
            assert bits(before) == bits(32. * after), (q, s)
            assert bits(before) == bits(weight(q) * v / 4), (q, s)
            checked += 1
    mismatches = []
    for a in range(65536):
        if a & 0x7f80 == 0x7f80:
            continue
        old = output(value(a), .25, 4.)
        new = output(value(a), 1./128, 128.)
        if old != new:
            mismatches.append(a)
    example = 0x0301
    assert output(value(example), .25, 4.) == example
    assert output(value(example), 1./128, 128.) == 0x0300
    permutation = (0, 2, 1, 3, 4, 6, 5, 7)
    lane_checks = operand_checks = 0
    for position in range(8):
        for s in range(256):
            raw = list(range(50,58))
            raw[position] = s
            after_swap = [raw[i] for i in permutation]
            for j in range(4):
                for i in range(2):
                    src = j // 2 * 4 + i * 2 + j % 2
                    assert after_swap[2*j+i] == raw[src]
                    lane_checks += 1
                    if math.isfinite(scale(raw[src])):
                        for q in range(16):
                            assert bits(weight(q) * scale(after_swap[2*j+i]) / 4) == bits(weight(q) * scale(raw[src]) / 4)
                            operand_checks += 1
    report = dict(finite_fp4_scale_pairs=checked, fold_original_operand_equals_32x_candidate=True,
                  single_nonzero_unit_weight_fixture=dict(K=4096, fp4_nibble=2, e4m3_byte=56,
                        activation="0x0301", parent="0x0301", public_fold="0x0300",
                        other_products=0, accumulator_model="binary32 without flush; candidate BF16 subnormal flushing gives zero instead"),
                  finite_bf16_activations_unit_weight=65280, pr103_mismatches=len(mismatches),
                  pr107_raw_index_checks=lane_checks, pr107_finite_weight_operand_checks=operand_checks,
                  pr107_domain="lane selection equal for arbitrary eight input words including both NaN codes; arithmetic and epilogue unchanged",
                  sources=["https://github.com/vllm-project/humming/pull/103","https://github.com/vllm-project/humming/pull/107"])
    (HERE/"cpu.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

