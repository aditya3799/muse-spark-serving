"""Independent NumPy forward packers, separated from inverse bridge indexing."""
import numpy as np


def wgs_forward(q, sf, gate_up):
    e, n, k = q.shape
    assert k == 4096 and sf.shape == (e, n, 256)
    tiles = e * (n // 256) * 64
    out = np.empty((tiles, 9216), np.uint8)
    idx = np.arange(2048, dtype=np.int64)[None, :]
    si = np.arange(1024, dtype=np.int64)[None, :]
    for first in range(0, tiles, 32):
        tile = np.arange(first, min(first + 32, tiles))[:, None]
        quad, step = tile % 4, tile // 4 % 16
        unit, expert = tile // 64 % (n // 256), tile // ((n // 256) * 64)
        base = unit * 256 + quad * 64 + idx // 128 % 4 * 16 + idx // 16 % 8
        col = step * 256 + idx // 512 * 64 + idx // 4 % 4 * 16 + idx % 4 * 2
        word = np.zeros((len(tile), 2048), np.uint32)
        for r in range(4):
            row = base + r % 2 * 8
            if gate_up:
                row = row // 16 * 8 + row % 8 + row % 16 // 8 * 4096
            for h in range(2):
                code = q[expert, row, col + r // 2 * 8 + h].astype(np.uint32)
                if r < 3:
                    word |= (code & 7) << np.uint32((6 + 3 * r + 16 * h) % 32)
                    word |= (code >> 3) << np.uint32((15 + 3 * r + 16 * h) % 32)
                else:
                    word |= ((code & 1) * 16 | (code >> 1) & 3 | code & 8) << np.uint32(16 * h)
        out[first:first + len(tile), :8192] = word.view(np.uint8)
        group = si // 8
        row = unit * 256 + quad * 64 + group // 8 % 4 * 16 + si % 2 * 8 + group % 8
        if gate_up:
            row = row // 16 * 8 + row % 8 + row % 16 // 8 * 4096
        scales = sf[expert, row, step * 16 + group // 32 * 4 + si // 2 % 4]
        out[first:first + len(tile), 8192:] = scales
    return out


def humming_forward(q, sf, wgmma):
    e, n, k = q.shape
    out = np.empty((e, k // 16, n * 2), np.uint32)
    # Forward tensor ownership derived from native weight_repack_nk, not inverse addresses.
    for expert in range(e):
        for first in range(0, n, 64):
            tile = q[expert, first:first + 64].reshape(4, 2, 8, k // 16, 2, 4, 2)
            axes = (3, 2, 5, 0, 4, 1, 6) if wgmma else (3, 2, 5, 0, 1, 4, 6)
            words = tile.transpose(axes).reshape(k // 16, 32, 4, 8)
            words = words[..., [0, 2, 4, 6, 1, 3, 5, 7]].astype(np.uint32)
            packed = np.bitwise_or.reduce(words << (np.arange(8, dtype=np.uint32) * 4), axis=-1)
            out[expert, :, first * 2:first * 2 + 128] = packed.reshape(k // 16, 128)
    perm = np.array([j + i for i in range(8) for j in range(0, 64, 8)])
    scales = sf.transpose(0, 2, 1).copy().reshape(-1, 64)[:, perm]
    return out, scales.reshape(e, k // 16, n)


def normal_forward(q, sf):
    e, n, k = q.shape
    codes = q[..., ::2] | (q[..., 1::2] << 4)
    # Each scale uint32 stores consecutive original groups, little endian.
    scales = sf.transpose(0, 2, 1).reshape(e, k // 64, 4, n).transpose(0, 1, 3, 2).copy()
    return codes, scales
