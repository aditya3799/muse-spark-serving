"""CPU-only address prototype: Humming 0.1.12 W4A16 -> v139 KPERM3 tiles.
No numerical conversion, CUDA imports, or persistent destination weight copy.
Source derivation: humming/include/humming/kernel/process.cuh and transform.py;
v139 _pack_codes_nat/_wgs_copy. Both source digests are in research_manifest.json.
"""

def humming_code_address(n, k, shape_n, transpose=True):
    assert shape_n % 64 == 0
    lane = (n % 8) * 4 + (k % 8) // 2
    i, j = (n % 64) // 8, (k % 64) // 8
    a, b = (j % 2, i % 2) if transpose else (i % 2, j % 2)
    tmp = ((((j // 2) * 4 + i // 2) * 2 + a) * 2 + b) * 2 + k % 2
    outer, word_j, inner = tmp // 32, (tmp % 32) // 8, tmp % 8
    slot = inner // 2 + 4 * (inner % 2)
    word = (k // 64 * 4 + outer) * (2 * shape_n) + n // 64 * 128 + lane * 4 + word_j
    return word, 4 * slot


def humming_scale_address(n, k_group, shape_n):
    return k_group * shape_n + n // 64 * 64 + n % 8 * 8 + n % 64 // 8


def donor_bit_position(register, upper, bit, mul=False):
    o = (6, 7, 8, 15)[bit]
    if mul:
        return (o + 16 * upper + (0, 3, 6)[register]) % 32 if register < 3 else (4, 0, 1, 3)[bit] + 16 * upper
    delta = (0, 28, 22, 18 if bit == 2 else 26)[register]
    return (o + 16 * upper + delta) % 32


def physical_row(logical_row, gate_up, intermediate=4096):
    if not gate_up:
        return logical_row
    return logical_row // 16 * 8 + logical_row % 8 + (intermediate if logical_row % 16 >= 8 else 0)


def wgs_word_coordinates(unit, stage, q, grp, warp, lane, word_j, gate_up, intermediate=4096):
    # KPERM3 destination new lane t is old four-step j; new word j is old lane t.
    g, old_j = lane // 4, lane % 4
    step = stage * 16 + grp * 4 + old_j
    for h, quarter in ((0, 0), (1, 0), (0, 1), (1, 1)):
        row = physical_row(unit * 256 + q * 64 + warp * 16 + g + 8 * h, gate_up, intermediate)
        for upper in (0, 1):
            yield row, step * 16 + quarter * 8 + word_j * 2 + upper


def translate_word(read_code, unit, stage, q, grp, warp, lane, word_j, gate_up=False, mul=False, intermediate=4096):
    out = 0
    for ix, (row, k) in enumerate(wgs_word_coordinates(unit, stage, q, grp, warp, lane, word_j, gate_up, intermediate)):
        code = read_code(row, k)
        for bit in range(4):
            out |= ((code >> bit) & 1) << donor_bit_position(ix // 2, ix % 2, bit, mul)
    return out


def translate_stage(codes, scales, shape_n, unit, stage, gate_up=False, mul=False, transpose=True):
    """One 256-row x256-K tile, 36KiB. Python diagnostic, not runtime implementation."""
    out = bytearray(36864)
    def read_code(row, k):
        word, shift = humming_code_address(row, k, shape_n, transpose)
        return (int.from_bytes(codes[4*word:4*word+4], 'little') >> shift) & 15
    for q in range(4):
        for grp in range(4):
            for warp in range(4):
                for lane in range(32):
                    for j in range(4):
                        word = translate_word(read_code, unit, stage, q, grp, warp, lane, j, gate_up, mul)
                        off = q * 9216 + (((grp * 4 + warp) * 32 + lane) * 4 + j) * 4
                        out[off:off+4] = word.to_bytes(4, 'little')
                for g in range(8):
                    for j in range(4):
                        for h in range(2):
                            row = physical_row(unit*256+q*64+warp*16+g+8*h, gate_up)
                            off = q*9216+8192+(((grp*4+warp)*8+g)*4+j)*2+h
                            out[off] = scales[humming_scale_address(row, stage*16+grp*4+j, shape_n)]
    return out
