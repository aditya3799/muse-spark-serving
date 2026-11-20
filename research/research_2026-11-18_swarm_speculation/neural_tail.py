"""Replay seven neural draft tokens with a causal 24-token history extension.

Synthetic neural candidates are a sensitivity experiment. The frozen policy sees
only the anchor, seven proposed IDs, available request budget and the live index.
The oracle outputs only construct simulated neural chains and score acceptance.
"""

import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit
import pyarrow.parquet as pq
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent / "research_2026-11-18_frontier"
EXPECTED_SHA = "ac2f41ef2444dec82fdc90e334f80543f298535005fe78aec7e8311ef86f84d0"


def trajectories(name):
    if name == "muse_joined":
        sys.path.insert(0, str(FRONTIER))
        from muse_records import samples

        yield from samples(joined=True)
    elif name == "openhands":
        for batch in pq.ParquetFile(FRONTIER / "openhands.parquet").iter_batches(
            columns=["messages"], batch_size=16
        ):
            for sample in batch.to_pylist():
                yield sample["messages"]
    else:
        for off in (1000, 5000, 12000, 20000):
            for row in json.loads(
                (HERE / f"traces_{off}.json").read_text(encoding="utf8")
            )["rows"]:
                yield json.loads(row["row"]["messages"])


def replay_features(messages, tok):
    # A key enters the index only after every one of its 24 source continuations
    # is in the past, and neither the key nor its tail spans a message boundary.
    history = []
    index = {}
    alternative = {}
    for msg in messages:
        history.append(-1)
        ids = tok.encode(msg["content"], add_special_tokens=False).ids
        assistant = msg["role"] == "assistant"
        features = np.zeros((len(ids), 3), np.uint8) if assistant else None
        for p, actual in enumerate(ids):
            if assistant and p > 0 and len(ids) - p >= 32:
                anchor = history[-1]
                true_key = (anchor, *ids[p : p + 7])
                old = index.get(true_key)
                if old is not None:
                    matched = 0
                    while (
                        matched < 24 and history[old + matched] == ids[p + 7 + matched]
                    ):
                        matched += 1
                    # zero is no source; 1 encodes an emitted correct bonus with
                    # no matched extension, which is already given by the base.
                    features[p, 0] = matched + 1
                fallback = alternative.get(anchor)
                perturbed_mask = 0
                donor_mask = 0
                for c in range(1, 8):
                    # c-1 correct, first incorrect at c, then fixed old-context
                    # filler; never consume real post-rejection suffix IDs.
                    right = ids[p + c - 1]
                    wrong = anchor if right != anchor else (anchor + 1) % 200000
                    fake = true_key[:c] + (wrong,) * (8 - c)
                    if fake in index:
                        perturbed_mask |= 1 << (c - 1)
                    if (
                        fallback is not None
                        and fallback[:c] == true_key[:c]
                        and fallback[c] != right
                    ):
                        # This valid but wrong alternative is a plausible
                        # secondary mode and *does* trigger a wasted long verify.
                        donor_mask |= 1 << (c - 1)
                features[p, 1] = perturbed_mask
                features[p, 2] = perturbed_mask | donor_mask
            history.append(actual)
            if p + 1 >= 32:
                donor_end = len(history) - 24
                key = tuple(history[donor_end - 8 : donor_end])
                index[key] = donor_end
                alternative[key[0]] = key
        if assistant and len(ids):
            yield features


def feature_data(name, tok):
    turns = []
    groups = []
    trace_n = 0
    for group, sample in enumerate(trajectories(name)):
        trace_n = group + 1
        for turn in replay_features(sample, tok):
            turns.append(turn)
            groups.append(group)
    ends = np.cumsum([len(t) for t in turns])
    return np.concatenate(turns), ends, np.array(groups), trace_n


@njit
def simulate(data, ends, groups, traces, uf, ul, pbase, odds, cdf, policy):
    # columns: steps, full draws, generated, wide, wasted wide with early
    # rejection, matched extra tokens, matched-key eligible visited, omitted near tail.
    out = np.zeros((traces, 8), np.int64)
    begin = 0
    for t in range(len(ends)):
        end = ends[t]
        i = begin
        group = groups[t]
        while i < end:
            remain = end - i
            corr = odds if data[i, 0] > 0 else 1.0
            p = pbase * corr / (1 - pbase + pbase * corr) if pbase > 0 else 0.0
            full = uf[i] < p
            c = 8
            if not full:
                c = 1
                while c < 7 and ul[i] >= cdf[c - 1]:
                    c += 1
            # Only condition on target oracle to *evaluate* the neural hit.
            # Per-mode features represent a concrete predicted seven-ID chain.
            wide = False
            gain = 0
            if policy and remain >= 32:
                if full:
                    wide = data[i, 0] > 0
                    if wide:
                        gain = int(data[i, 0]) - 1
                elif policy == 3:
                    wide = True
                elif policy == 1 or policy == 2:
                    wide = bool(int(data[i, policy]) & (1 << (c - 1)))
            count = min(remain, c + gain)
            out[group, 0] += 1
            out[group, 1] += full and remain >= 8
            out[group, 2] += count
            out[group, 3] += wide
            out[group, 4] += wide and not full
            out[group, 5] += gain
            out[group, 6] += data[i, 0] > 0
            out[group, 7] += remain < 32
            i += count
        begin = end
    return out


def calibrate(data, ends, groups, n, uf, ul, want_p, odds):
    def cdf(slope):
        w = np.exp(np.arange(7) * slope)
        return np.cumsum(w) / sum(w)

    # Simultaneously match mean emitted tokens and full draw rate. Calibrate on
    # baseline visited positions, including end-of-turn truncation effects.
    p = want_p
    slope = 0.0
    for _ in range(4):
        if want_p:
            lo, hi = 0.0, 1.0
            for j in range(14):
                p = (lo + hi) / 2
                a = simulate(data, ends, groups, n, uf, ul, p, odds, cdf(slope), 0).sum(
                    axis=0
                )
                if a[1] / a[0] < want_p:
                    lo = p
                else:
                    hi = p
        lo, hi = -4.0, 2.0
        for j in range(14):
            slope = (lo + hi) / 2
            a = simulate(data, ends, groups, n, uf, ul, p, odds, cdf(slope), 0).sum(
                axis=0
            )
            if a[2] / a[0] < 4.0:
                lo = slope
            else:
                hi = slope
    return p, cdf(slope)


def main(name):
    path = FRONTIER / "muse_tokenizer.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA
    data, ends, groups, n = feature_data(name, Tokenizer.from_file(str(path)))
    rng = np.random.default_rng(183109)
    uf, ul = (
        rng.random(len(data), dtype=np.float32),
        rng.random(len(data), dtype=np.float32),
    )
    bootstrap = rng.integers(n, size=(1000, n))
    report = {
        "dataset": name,
        "trajectories": n,
        "output_tokens": len(data),
        "turns": len(ends),
        "tokenizer_sha256": EXPECTED_SHA,
        "cases": [],
    }
    for full_p in (0.0, 0.10, 0.25, 0.40):
        for odds in (1.0,) if full_p == 0 else (1.0, 4.0):
            p, cdf = calibrate(data, ends, groups, n, uf, ul, full_p, odds)
            baseline = simulate(data, ends, groups, n, uf, ul, p, odds, cdf, 0)
            b = baseline.sum(axis=0)
            case = {
                "target_full_probability": full_p,
                "full_odds_multiplier_on_true_key": odds,
                "observed_base_full_commits": float(b[1] / b[0]),
                "observed_base_mean": float(b[2] / b[0]),
                "base_steps": int(b[0]),
                "policies": {},
            }
            for policy, label in [
                (1, "perturbed"),
                (2, "historical_wrong_or_perturbed"),
                (3, "all_partial_wide_upper_cost"),
            ]:
                counts = simulate(data, ends, groups, n, uf, ul, p, odds, cdf, policy)
                x = counts.sum(axis=0)
                assert x[2] == b[2] == len(data)
                if full_p == 0:
                    assert x[0] == b[0] and x[5] == 0
                draws = baseline[:, 0][bootstrap].sum(1) / (
                    counts[:, 0] + 0.25 * counts[:, 3]
                )[bootstrap].sum(1)
                case["policies"][label] = dict(
                    steps=int(x[0]),
                    wide=int(x[3]),
                    wasted_wide_partial=int(x[4]),
                    matched_extra_tokens=int(x[5]),
                    target_forward_count_ratio=float(b[0] / x[0]),
                    relative_target_rows=float((x[0] * 8 + x[3] * 24) / (b[0] * 8)),
                    wide_cost_break_even=float(1 + (b[0] - x[0]) / x[3])
                    if x[3]
                    else None,
                    ratio_at_wide_cost_1p25=float(b[0] / (x[0] + 0.25 * x[3])),
                    ratio_at_wide_cost_1p5=float(b[0] / (x[0] + 0.5 * x[3])),
                    trace_bootstrap_1p25_interval=np.quantile(
                        draws, [0.025, 0.975]
                    ).tolist(),
                )
            report["cases"].append(case)
            print(
                name,
                full_p,
                odds,
                round(case["observed_base_mean"], 4),
                round(case["observed_base_full_commits"], 4),
                {
                    k: round(v["ratio_at_wide_cost_1p25"], 4)
                    for k, v in case["policies"].items()
                },
                flush=True,
            )
    (HERE / f"neural_tail_{name}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
