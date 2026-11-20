"""Counterfactual continuous-batch decode under a fixed device graph row budget.

Policies use only past tokens and the declared output budget. The offline oracle
knows true future only when scoring an already selected chain. Width shortening
uses synthetic neural acceptance lengths because paired model logits are absent.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from numba import njit
from tokenizers import Tokenizer
from cross_validate_lookup import dataset, per_turn

HERE = Path(__file__).resolve().parent


@njit
def simulate(matches, order, offsets, neural, batch, per_row, minimum, mode):
    # mode 0 uses every ordinary block; 1 reallocates rows in one static graph.
    pos = np.zeros(batch, np.int64)
    stop = np.zeros(batch, np.int64)
    step = 0
    cursor = 0
    active_steps = 0
    generated = 0
    wide = 0
    trimmed = 0
    charged_rows = 0
    for _ in range(len(matches)+1):
        active = 0
        for r in range(batch):
            if pos[r] == stop[r] and cursor < len(order):
                turn = order[cursor]
                cursor += 1
                pos[r], stop[r] = offsets[turn], offsets[turn+1]
            active += pos[r] < stop[r]
        if active == 0:
            break
        widths = np.full(batch, minimum if mode else 8, dtype=np.int64)
        is_copy = np.zeros(batch,np.bool_)
        if mode:
            budget = active * (per_row-minimum)
            # Rotate equal-priority request order. Commit to each extension
            # without inspecting the target response or the future match length.
            for j in range(batch):
                r = (step+j) % batch
                i = pos[r]
                if stop[r]-i < 32 or matches[i] == 255:
                    continue
                for proposed in (32,24,16):
                    if proposed-minimum <= budget:
                        is_copy[r] = True
                        widths[r] = proposed
                        budget -= proposed-minimum
                        break
            # Return unused rows to the ordinary DFlash chain on fallback rows.
            for j in range(batch):
                r = (step+j) % batch
                if pos[r] < stop[r] and not is_copy[r]:
                    extra = min(8-minimum,budget)
                    widths[r] += extra
                    budget -= extra
        for r in range(batch):
            i = pos[r]
            if i == stop[r]:
                continue
            accept = int(matches[i])+1 if is_copy[r] else int(neural[i])
            take = min(accept,widths[r],stop[r]-i)
            pos[r] += take
            generated += take
            wide += is_copy[r]
            trimmed += widths[r] < 8
        active_steps += active
        charged_rows += active * (per_row if mode else 8)
        step += 1
    assert generated == len(matches)
    return step,charged_rows,wide,trimmed,active_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus",choices=["swesmith","openhands","muse_joined","muse_split"])
    corpus = p.parse_args().corpus
    tok = Tokenizer.from_file(str(HERE/"muse_tokenizer.json"))
    turns = [np.array([v if k>=8 and avail>=31 else 255 for k,v,avail in t],dtype=np.uint8)
             for trace in dataset(corpus) for t in per_turn(trace,tok) if t]
    offsets = np.cumsum([0,*[len(t) for t in turns]])
    matches = np.concatenate(turns)
    collected = []
    for seed in (7,71,919):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(turns))
        # Same per-position synthetic neural prediction across experiments.
        # Compare deterministic and dispersed equal-mean acceptances.
        for spread in (False,True):
            neural = (rng.choice(np.array([1,2,3,4,5,6,7],np.uint8),size=len(matches))
                      if spread else np.full(len(matches),4,dtype=np.uint8))
            for batch in (1,4,16,32):
                base=simulate(matches,order,offsets,neural,batch,8,8,0)
                for per_row in (8,12,16):
                    trial=simulate(matches,order,offsets,neural,batch,per_row,4,1)
                    collected.append(dict(seed=seed,spread=spread,batch=batch,per_row=per_row,
                        baseline_rounds=base[0],candidate_rounds=trial[0],round_ratio=base[0]/trial[0],
                        paid_target_row_ratio=trial[1]/base[1],copy_row_fraction=trial[2]/trial[4],
                        shortened_fallback_fraction=trial[3]/trial[4]))
    data=dict(corpus=corpus,tokens=len(matches),turns=len(turns),results=collected,
              scope="fixed charged rows per live request per target pass, excludes GPU time, prefill, draft work and graph rounding at batch tail")
    (HERE/f"{corpus}_fixed_rows.json").write_text(json.dumps(data,indent=2))
    print(corpus,[(r["batch"],round(r["round_ratio"],3)) for r in collected if r["seed"]==7 and r["spread"] and r["per_row"]==8])
    

if __name__ == "__main__":
    main()
