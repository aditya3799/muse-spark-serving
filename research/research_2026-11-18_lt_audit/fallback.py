"""Fail-closed sketch for calibrated output-shard plans on replicated input and weights.

The TP group must visit the same registered key on every rank. Group members negotiate
during initialization. configure() must finish before any CUDA graph is captured.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalPlan:
    run: Callable
    # Deployment supplies context (software identity, SM count, runtime math flags,
    # layout/alignment, and the available, stream-owned workspace) as a key.


class ReplicatedProjection:
    def __init__(self, native_linear, all_ranks_ready, gather_shards):
        self.native = native_linear  # Keep replicated native weights for every fallback.
        self.all_ranks_ready = all_ranks_ready
        self.gather_shards = gather_shards
        self.plans = {}

    def configure(self, key, local_plan):
        # All ranks call this once in the same startup order, including unsupported ranks.
        agreed = self.all_ranks_ready(local_plan is not None)
        self.plans[key] = local_plan if agreed else None

    def __call__(self, x, key):
        plan = self.plans.get(key)
        if plan is None:
            return self.native(x)
        # Graph-safe exact validation. An empirically good plan can still diverge
        # on future data or firmware. Returning native bits on a disagreement is
        # unconditional. The validation costs a full projection in this mode.
        baseline = self.native(x)
        answer = self.gather_shards(plan.run(x))
        if answer.dtype != baseline.dtype or answer.shape != baseline.shape:
            raise RuntimeError("output-shard registration disagrees with native output")
        if baseline.dtype != torch.bfloat16:
            raise RuntimeError("only the BF16 storage representation was validated")
        original_bits = baseline.view(torch.int16)
        candidate_bits = answer.view(torch.int16)
        use_candidate = (original_bits == candidate_bits).all()
        return torch.where(use_candidate, candidate_bits, original_bits).view(torch.bfloat16)


def self_test():
    native = torch.nn.Linear(32,16,bias=False,dtype=torch.bfloat16)
    agreed = [False]
    entered = []
    def gather(y):
        entered.append(True)
        return y
    def bad_result(x):
        bits = native(x).detach().view(torch.int16).clone()
        bits[0,0] ^= 1
        return bits.view(torch.bfloat16)
    run = ReplicatedProjection(native,lambda local: local and agreed[0],gather)
    x = torch.randn(8,32,dtype=torch.bfloat16)
    expected = native(x).view(torch.int16)
    for key,plan in [("absent",None),("other rank missing",LocalPlan(bad_result))]:
        run.configure(key,plan)
        assert torch.equal(run(x,key).view(torch.int16), expected)
    assert torch.equal(run(x,"never registered").view(torch.int16), expected)
    assert not entered
    agreed[0] = True
    for key,plan in [("differs",LocalPlan(bad_result)),("matches",LocalPlan(native))]:
        run.configure(key,plan)
        assert torch.equal(run(x,key).view(torch.int16), expected)
    assert len(entered)==2
    # Signed zero and distinct NaN payloads expose comparisons using numerical ==.
    def special(_):
        return torch.tensor([-32768,0,32641,32767],dtype=torch.int16).view(torch.bfloat16)
    run = ReplicatedProjection(special,lambda ready:ready,lambda y:y)
    run.configure("nonfinite",LocalPlan(lambda x:torch.zeros_like(special(x))))
    assert torch.equal(run(x,"nonfinite").view(torch.int16),special(x).view(torch.int16))
    print("unconfigured, missing plan, other rank missing, one-bit disagreement, exact match, signed zero and NaN payloads passed")


if __name__=="__main__":
    self_test()

