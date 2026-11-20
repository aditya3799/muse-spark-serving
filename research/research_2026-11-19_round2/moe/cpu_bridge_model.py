"""NumPy bridge model for independent forward-oracle comparison; no Torch import."""
import numpy as np


def read_code(src, e, n, k, width, up):
    p = (n % 4096 // 8 * 16 + n % 8 + n // 4096 * 8) if up else n
    tile = ((e * (width // 256) + p // 256) * 16 + k // 256) * 4 + p % 256 // 64
    index = k % 256 // 64 * 512 + p % 64 // 16 * 128 + p % 8 * 16 + k % 64 // 16 * 4 + k % 8 // 2
    word = src.reshape(-1).view(np.uint32)[tile * 2304 + index]
    r = p // 8 % 2 + k // 8 % 2 * 2
    h = k % 2
    mag = (word >> ((6 + 3 * r + 16 * h) % 32).astype(np.uint32)) & 7
    sign = ((word >> ((15 + 3 * r + 16 * h) % 32).astype(np.uint32)) & 1) * 8
    half = word >> (16 * h).astype(np.uint32)
    special = (half >> 4) & 1 | (half & 3) * 2 | half & 8
    return np.where(r == 3, special, mag | sign).astype(np.uint8)


def read_scale(src, e, n, group, width, up):
    p = (n % 4096 // 8 * 16 + n % 8 + n // 4096 * 8) if up else n
    tile = ((e * (width // 256) + p // 256) * 16 + group // 16) * 4 + p % 256 // 64
    index = group % 16 // 4 * 256 + p % 64 // 16 * 64 + p % 8 * 8 + group % 4 * 2 + p // 8 % 2
    return src.reshape(-1)[tile * 9216 + 8192 + index]


def convert(src, scratch, e, n, layout):
    total = e * n * 2304
    assert scratch.dtype == np.uint8 and len(scratch) >= total
    up = n == 8192
    size = e * n * 2048
    output = scratch[:size]
    sf = scratch[size:total]
    for expert in range(e):
        if layout == 'normal':
            for first in range(0, n * 2048, 65536):
                z = np.arange(first, min(first + 65536, n * 2048), dtype=np.int64)
                row, col = z // 2048, z % 2048 * 2
                output[expert * n * 2048 + z] = read_code(src, expert, row, col, n, up) | (read_code(src, expert, row, col + 1, n, up) << 4)
        else:
            words = output.view(np.uint32)
            for first in range(0, n * 512, 16384):
                z = np.arange(first, min(first + 16384, n * 512), dtype=np.int64)
                kb, nb, lane, a = z // (n * 2), z % (n * 2) // 128, z % 128 // 4, z % 4
                word = np.zeros(len(z), np.uint32)
                for i in range(2):
                    for j in range(2):
                        pair = j * 2 + i if layout == 'humming_wgmma' else i * 2 + j
                        row = nb * 64 + a * 16 + i * 8 + lane // 4
                        col = kb * 16 + j * 8 + lane % 4 * 2
                        word |= read_code(src, expert, row, col, n, up).astype(np.uint32) << (pair * 4)
                        word |= read_code(src, expert, row, col + 1, n, up).astype(np.uint32) << (pair * 4 + 16)
                words[expert * n * 512 + z] = word
        for first in range(0, n * 256, 65536):
            z = np.arange(first, min(first + 65536, n * 256), dtype=np.int64)
            if layout == 'normal':
                row, group = z // 4 % n, z // (4 * n) * 4 + z % 4
            else:
                p = z % n
                row, group = p // 64 * 64 + p % 8 * 8 + p % 64 // 8, z // n
            sf[expert * n * 256 + z] = read_scale(src, expert, row, group, n, up)
    return output, sf
