"""Exercise worker-facing frontier conventions and route decisions without a GPU."""

import sys

import torch

from suffix_routing import DFlashCopyRouter
from suffix_store import SuffixStore


def verify(device):
    store = SuffixStore(3, 1024, device=device)
    router = DFlashCopyRouter(store)
    key = list(range(10, 28))
    history = [3, 4, *key, *range(100, 170), *key]
    first = router.finish_prefill(2, "first", history[:-1], history[-1])
    miss = router.finish_prefill(0, "unrelated", [5, 6], 7)
    last = router.finish_prefill(1, "third", history[:-1], history[-1])
    slots = [last.slot, miss.slot, first.slot]
    epochs = [last.epoch, miss.epoch, first.epoch]
    kv = [len(history) - 1, 2, len(history) - 1]
    bonus = [history[-1], 7, history[-1]]

    result = router.route(slots, epochs, kv, bonus, [64, 64, 64], [True] * 3)
    assert result.copy_rows.tolist() == [True, False, True]
    assert result.verify_widths.tolist() == [32, 8, 32]
    assert result.uniform_width.item() == 8
    assert result.proposals[0].tolist() == [history[-1], *range(100, 131)]
    assert result.proposals[0].equal(result.proposals[2])

    hits = [first, last]
    result = router.route(
        [h.slot for h in hits],
        [h.epoch for h in hits],
        [kv[0]] * 2,
        [bonus[0]] * 2,
        [32] * 2,
        [True] * 2,
    )
    assert result.all_copy.item() and result.uniform_width.item() == 32
    result = router.route(slots, epochs, kv, bonus, [40, 64, 31], [False, True, True])
    assert result.verify_widths.tolist() == [8] * 3
    result = router.route([], [], [], [], [], torch.empty(0, dtype=torch.bool))
    assert result.uniform_width.item() == 8

    output = [[100, 401, 0], [8, 0, 0], [100, 101, 700]]
    accepted = router.accept(slots, epochs, kv, output, [2, 1, 3])
    assert accepted.tolist() == [True, True, True]
    assert store.tokens[2, len(history) : len(history) + 3].tolist() == [100, 101, 700]
    assert store.frontier.tolist() == [4, len(history) + 2, len(history) + 3]
    assert not router.accept(slots, epochs, kv, output, [2, 1, 3]).any().item()
    print(
        f"Passed {device} prefill, mixed/all/empty routing, exact commit and retry checks."
    )


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else "cpu")
