"""CPU-only exact *token-state* contract, not numerical model equivalence.
No serving code is imported or modified. Source-extracted CPU accept is tested
separately when torch is available. Variable verify width remains held.
"""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRC = ROOT / 'sglang/python/sglang/kernels/ops/speculative/dflash.py'
WIDTHS = (1, 2, 4, 6, 8)

def accept(candidates, predictions):
    assert len(candidates) == len(predictions) and candidates
    a = 0
    while a < len(candidates) - 1 and candidates[a + 1] == predictions[a]:
        a += 1
    bonus = predictions[a]
    return a, a + 1, bonus, candidates[1:a + 1] + [bonus]

def target(prefix):
    # Causal deterministic finite-vocabulary oracle; uses the complete prefix.
    h = 13
    for token in prefix:
        h = (h * 31 + token + 1) % 65521
    return h % 11

def propose(full_prefix, seed, iteration):
    # Always produces seven proposals, independently of verify width.
    rng = random.Random(seed * 100003 + iteration)
    current = list(full_prefix)
    proposals = []
    for i in range(7):
        exact = target(current)
        token = exact if rng.randrange(4) else (exact + 1 + i) % 11
        proposals.append(token)
        current.append(token)
    return proposals

def run(seed, widths, length=120):
    prompt = [seed % 11, (seed + 5) % 11]
    anchor = target(prompt)
    emitted = [anchor]
    committed = list(prompt)
    target_kv = dict(enumerate(committed))
    draft_kv = dict(enumerate(committed))
    iteration = 0
    steps = []
    while len(emitted) < length:
        width = widths[iteration % len(widths)]
        assert 1 <= width <= 8
        proposals = propose(committed + [anchor], seed, iteration)
        full = [anchor] + proposals
        candidates = full[:width]
        predictions = [target(committed + candidates[:i + 1]) for i in range(width)]
        a, count, next_anchor, output = accept(candidates, predictions)
        before = len(committed)
        # Target may speculatively write every verified row. Stale rejected rows
        # remain outside the read extent, and are overwritten by later verifies.
        for i, token in enumerate(candidates):
            target_kv[before + i] = token
        # Draft context writer commits exactly anchor + accepted proposals.
        for i in range(count):
            draft_kv[before + i] = candidates[i]
        committed += candidates[:count]
        emitted += output
        anchor = next_anchor
        for pos, token in enumerate(committed):
            assert target_kv[pos] == token and draft_kv[pos] == token
        assert committed + [anchor] == prompt + emitted
        assert len(committed) == before + count
        assert output[-1] == anchor and len(output) == count
        assert full[0] == candidates[0] and len(full) == 8
        steps.append((width, a, count))
        iteration += 1
    expected = list(prompt)
    for _ in range(len(emitted)):
        expected.append(target(expected))
    assert prompt + emitted == expected
    return emitted[:length], steps

def main():
    report = {'status': 'CPU_TOKEN_STATE_ONLY', 'widths': WIDTHS,
              'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
              'accept_cases': 0, 'sequence_cases': 0, 'state_steps': 0,
              'negative_controls': {},
              'limits': ['No real model arithmetic or graph replay',
                         'No performance measurement',
                         'No challenge rule eligibility assertion',
                         'Full target shape-dependent numerical equivalence unresolved']}
    # Exhaust every binary match/mismatch pattern, not just first-mismatch index.
    for width in WIDTHS:
        for mask in itertools.product((False, True), repeat=width - 1):
            for anchor in (0, 4, 17):
                candidates = [anchor] + list(range(30, 30 + width - 1))
                predicted = [candidates[i + 1] if ok else 100 + i
                             for i, ok in enumerate(mask)] + [999]
                a, c, bonus, output = accept(candidates, predicted)
                expected_a = next((i for i, ok in enumerate(mask) if not ok), width - 1)
                assert (a, c, bonus) == (expected_a, expected_a + 1, predicted[expected_a])
                assert output == candidates[1:c] + [bonus]
                report['accept_cases'] += 1
    schedules = [(w,) for w in WIDTHS] + [(2, 8, 4, 6), (8, 1, 6, 2, 4)]
    for seed in range(100):
        reference, _ = run(seed, (8,))
        for schedule in schedules:
            actual, steps = run(seed, schedule)
            assert actual == reference
            report['sequence_cases'] += 1
            report['state_steps'] += len(steps)
    # Flat-prefix slicing and output-stride8 are wrong for multi-request V<8.
    bs, v = 3, 4
    rows = [[100 * row + col for col in range(8)] for row in range(bs)]
    correct = [x for row in rows for x in row[:v]]
    wrong = sum(rows, [])[:bs * v]
    assert correct != wrong
    report['negative_controls']['flat_slice_corrupts_request_1'] = [correct[v], wrong[v]]
    slots = sum([[1000 * row + col for col in range(v)] for row in range(bs)], [])
    assert slots[v] != slots[8]
    report['negative_controls']['scheduler_stride8_reads_request2_for_request1'] = [slots[v], slots[8]]
    # The emitted bonus has no KV until the next iteration: storing emitted
    # output at the current KV positions drops the anchor.
    candidates, predicted = [10, 20, 30, 40], [20, 99, 98, 97]
    a, c, bonus, output = accept(candidates, predicted)
    assert candidates[:c] != output and candidates[:c] == [10, 20] and output == [20, 99]
    report['negative_controls']['output_tokens_are_not_committed_input_rows'] = {'kv': candidates[:c], 'output': output}
    # A real target that changes an argmax with geometry defeats the proof.
    narrow = accept([10, 20], [21, 99])[-1]
    wide = accept([10, 20, 30, 40], [20, 30, 40, 99])[-1]
    assert narrow[0] != wide[0]
    report['negative_controls']['shape_sensitive_target_breaks_token_equivalence'] = [narrow[0], wide[0]]
    try:
        import torch
        tree = ast.parse(SRC.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_dflash_correct_drafts_and_bonus')
        namespace = {'torch': torch, 'Tuple': tuple}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC), 'exec'), namespace)
        pinned = namespace[fn.name]
        cases = 0
        for width in WIDTHS:
            for mask in itertools.product((False, True), repeat=width - 1):
                cand = [7] + list(range(30, 30 + width - 1))
                pred = [cand[i + 1] if ok else 100 + i for i, ok in enumerate(mask)] + [999]
                a, _, b, _ = accept(cand, pred)
                aa, bb = pinned(candidates=torch.tensor([cand]), target_predict=torch.tensor([pred]))
                assert aa.item() == a and bb.item() == b
                cases += 1
        report['source_extracted_cpu_accept_cases'] = cases
    except ImportError:
        report['source_extracted_cpu_accept_cases'] = 'NOT_RUN_TORCH_UNAVAILABLE'
    (HERE / 'VERIFY_PREFIX_CPU_CONTRACT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
