"""Research-only unchanged-draft8 / shorter-target-prefix protocol. No model integration."""
from dataclasses import dataclass
from typing import Tuple

DRAFT_WIDTH = 8
SUPPORTED = (1, 2, 4, 6, 8)

def pack_prefix(rows, width):
    if width not in SUPPORTED or any(len(row) != DRAFT_WIDTH for row in rows):
        raise ValueError("Expected full draft8 rows and supported verify width")
    return tuple(tuple(row[:width]) for row in rows)

@dataclass(frozen=True)
class Commit:
    accepted: int
    committed: int
    bonus: int
    emitted: Tuple[int, ...]
    input_tokens: Tuple[int, ...]
    new_seq_len: int

def accept_prefix(candidates, target_top1, prefix_len):
    if not candidates or len(candidates) != len(target_top1):
        raise ValueError("Target must predict exactly one token per verified input")
    accepted = 0
    for proposed, target in zip(candidates[1:], target_top1[:-1]):
        if proposed != target:
            break
        accepted += 1
    committed = accepted + 1
    bonus = target_top1[accepted]
    return Commit(accepted, committed, bonus,
                  tuple(candidates[1:committed]) + (bonus,),
                  tuple(candidates[:committed]), prefix_len + committed)

class AuxBuffer:
    """Models one graph-owned output buffer reused by every target forward."""
    def __init__(self):
        self.generation = 0
        self.rows = ()
    def write(self, rows):
        self.generation += 1
        self.rows = tuple(rows)
        return self.generation
    def consume(self, generation, committed):
        if generation != self.generation:
            raise RuntimeError("Target auxiliary output was overwritten before draft materialization")
        if not 1 <= committed <= len(self.rows):
            raise ValueError("Cannot consume unverified or uncommitted rows")
        return self.rows[:committed]

def target_next(history):
    # Causal toy oracle, deliberately depends on all previous tokens.
    return (sum((i + 3) * x for i, x in enumerate(history)) + 7 * len(history) + 11) % 31

def greedy(history, count):
    h = list(history)
    out = []
    for _ in range(count):
        out.append(target_next(h)); h.append(out[-1])
    return out

def full_draft8(history, first_wrong):
    # Construct the full bidirectional-draft interface BEFORE choosing verify K.
    # Toy proposal policy, not a substitute for the trained draft model.
    row = [history[-1]]
    h = list(history)
    for i in range(7):
        token = target_next(h)
        if i >= first_wrong:
            token = (token + 1 + i) % 31
        row.append(token); h.append(token)
    return tuple(row)

def verify(history_before_anchor, candidates):
    h = list(history_before_anchor)
    top1, aux = [], []
    for token in candidates:
        h.append(token)
        top1.append(target_next(h))
        aux.append(tuple(h))  # Represents hidden features for this input, not next bonus.
    return tuple(top1), tuple(aux)

def run_request(prompt, count, widths, mismatch_pattern, eos=None, physical=None):
    if not prompt or count < 1:
        raise ValueError("Nonempty prompt and positive generation budget required")
    # Prompt except final anchor is already materialized. The anchor is pending.
    committed_history = list(prompt[:-1])
    anchor = prompt[-1]
    generated = []
    target_slots = {} if physical is None else physical
    draft_slots = {i: tuple(prompt[:i+1]) for i in range(len(prompt)-1)}
    aux_buffer = AuxBuffer()
    rows = 0; cycles = 0; reused_uncommitted = 0; all_accepted = 0
    while len(generated) < count:
        history = committed_history + [anchor]
        full = full_draft8(history, mismatch_pattern[cycles % len(mismatch_pattern)])
        width = widths[cycles % len(widths)]
        candidates = pack_prefix([full], width)[0]
        assert full[0] == anchor and len(full) == 8
        predictions, aux = verify(committed_history, candidates)
        base = len(committed_history)
        for i, value in enumerate(aux):
            if base+i in target_slots:
                reused_uncommitted += 1
            target_slots[base+i] = value
        generation = aux_buffer.write(aux)
        commit = accept_prefix(candidates, predictions, base)
        if commit.committed == width:
            all_accepted += 1
            assert commit.bonus == predictions[-1]
        # Eager consumption is required before any subsequent target invocation.
        committed_aux = aux_buffer.consume(generation, commit.committed)
        for i, value in enumerate(committed_aux):
            draft_slots[base+i] = value
        committed_history.extend(commit.input_tokens)
        for i in range(len(committed_history)):
            expected = tuple(committed_history[:i+1])
            if i >= len(prompt)-1:
                assert target_slots[i] == expected
            assert draft_slots[i] == expected
        # Bonus is pending: its KV and hidden cannot be used. A stale speculative
        # slot may physically exist there, but it is outside the live logical length.
        assert commit.new_seq_len == len(committed_history)
        anchor = commit.bonus
        remaining = count-len(generated)
        visible = list(commit.emitted[:remaining])
        done = False
        if eos is not None and eos in visible:
            visible = visible[:visible.index(eos)+1]; done = True
        generated.extend(visible)
        rows += width; cycles += 1
        if done:
            break
    reference = greedy(prompt, count)
    if eos is not None and eos in reference:
        reference = reference[:reference.index(eos)+1]
    assert generated == reference, (generated, reference)
    return dict(output=generated, target_rows=rows, cycles=cycles,
                reused_slots=reused_uncommitted, all_accepted=all_accepted,
                committed_length=len(committed_history), pending_bonus=anchor)
