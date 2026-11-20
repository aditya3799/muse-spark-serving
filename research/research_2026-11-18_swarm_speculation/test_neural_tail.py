"""Independent greedy-chain and causal-index checks for intact-prefix extension."""

import random

import numpy as np

from neural_tail import replay_features


def accept(candidates, teacher, prefix):
    live = tuple(prefix)
    emitted = []
    for draft in candidates[1:]:
        actual = teacher(live)
        emitted.append(actual)
        if draft != actual:
            return emitted
        live += (actual,)
    emitted.append(teacher(live))
    return emitted


class IdTokenizer:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, s, add_special_tokens=False):
        return self.Encoding([int(v) for v in s.split()])


def test():
    rng = random.Random(52713)
    checks = 0
    for _ in range(128):
        # History-sensitive serial teacher exercises arbitrary wrong branches.
        def teacher(ids):
            return (sum((i + 1) * t for i, t in enumerate(ids[-6:])) * 31 + 17) % 4093

        prefix = tuple(rng.randrange(4093) for _ in range(10))
        correct = []
        for j in range(33):
            correct += [teacher(prefix + tuple(correct))]
        for c in range(1, 9):
            chain = correct[:7]
            if c < 8:
                chain[c - 1] = (chain[c - 1] + 1) % 4093
                chain[c:] = [rng.randrange(4093) for _ in chain[c:]]
            base = accept([prefix[-1], *chain], teacher, prefix)
            assert base == correct[:c]
            for extra in (0, 1, 8, 23, 24):
                tail = correct[7:31]
                if extra < 24:
                    tail[extra] = (tail[extra] + 1) % 4093
                wide = accept([prefix[-1], *chain, *tail], teacher, prefix)
                expected = c if c < 8 else 8 + extra
                assert wide == correct[:expected] and wide[: len(base)] == base
                checks += 1
    tok = IdTokenizer()
    key = list(range(70, 78))
    tail = list(range(100, 124))
    target = key + tail + list(range(40))
    message = lambda role, t: {"role": role, "content": " ".join(str(v) for v in t)}
    valid = list(
        replay_features(
            [message("user", key + tail), message("assistant", target)], tok
        )
    )[0]
    assert valid[1, 0] == 25
    alone = list(replay_features([message("assistant", target)], tok))[0]
    assert alone[1, 0] == 0
    cut = list(
        replay_features(
            [message("user", key + tail[:23]), message("assistant", target)], tok
        )
    )[0]
    assert cut[1, 0] == 0
    bad = target[:]
    bad[8] += 1000
    bad_result = list(
        replay_features([message("user", key + tail), message("assistant", bad)], tok)
    )[0]
    assert bad_result[1, 0] == 1
    assert np.array_equal(valid[1, 1:], bad_result[1, 1:])
    print(
        "Passed",
        checks,
        "independent serial verifier comparisons and four causal donor checks.",
    )


if __name__ == "__main__":
    test()
