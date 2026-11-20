"""Spend a fixed target budget by confidence, and test what observations identify."""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "decode_selector"
BATCH = 4
DRAFT_SLOTS = 7
BUDGET = 12


class ConditionalAcceptance:
    """Reach-conditional binary labels. Later positions of rejected chains are censored."""

    def __init__(self, bins=8):
        self.bins = bins
        self.count = defaultdict(lambda: [0, 0])

    def key(self, width, regime, slot, confidence):
        band = min(self.bins - 1, int(max(0.0, min(1.0, confidence)) * self.bins))
        return width, regime, slot, band

    def observe(self, width, regime, confidences, verified, accepted, rejected):
        if not 0 <= accepted <= verified <= width == len(confidences):
            raise ValueError("inconsistent lengths")
        if rejected and accepted == verified:
            raise ValueError("cannot observe rejection of an unverified slot")
        for slot in range(accepted + int(rejected)):
            key = self.key(width, regime, slot, confidences[slot])
            bucket = self.count[key]
            bucket[0] += int(slot < accepted)
            bucket[1] += 1

    def conditional(self, width, regime, confidences, min_reached=4):
        result = []
        for slot, c in enumerate(confidences):
            yes, total = self.count.get(self.key(width, regime, slot, c), (0, 0))
            if total < min_reached:
                return None
            result.append((yes + 1) / (total + 2))
        return result


def prefix_budget(scores, budget):
    """Positive equal row costs: every chain's prefix survival is nonincreasing."""
    by_request = [list(itertools.accumulate(probs, lambda a, b: a * b)) for probs in scores]
    slots = [0] * len(scores)
    for _ in range(budget):
        eligible = [i for i, used in enumerate(slots) if used < len(scores[i])]
        i = max(eligible, key=lambda row: (by_request[row][slots[row]], -row))
        slots[i] += 1
    return slots


def emitted(widths, accepted):
    return len(widths) + sum(min(w, a) for w, a in zip(widths, accepted))


def main():
    record = json.loads((ROOT / "lifecycle_cpu.json").read_text(encoding="utf-8-sig"))
    case = []
    greedy_raw = []
    for step in record["fixtures"]:
        raw = [[step["selected_raw_softmax"][pos * BATCH + row] for pos in range(DRAFT_SLOTS)] for row in range(BATCH)]
        q = [[step["selected_draft_q"][pos * BATCH + row] for pos in range(DRAFT_SLOTS)] for row in range(BATCH)]
        for row in range(BATCH):
            if step["mode"] in ("none", "greedy") or (step["mode"] == "mixed" and row % 2):
                assert q[row] == [1.0] * DRAFT_SLOTS
                greedy_raw.extend(raw[row])
        widths = prefix_budget(raw, BUDGET)
        assert sum(widths) == BUDGET
        fixed = [BUDGET // BATCH] * BATCH
        # Preserve same 4 exact target-trace lengths, with no measured correlation imposed.
        pairs = [(emitted(widths, t), emitted(fixed, t)) for t in itertools.permutations([0, 1, 3, 7])]
        assert all(y == 11 for _, y in pairs)
        ratios = [x/y for x,y in pairs]
        case.append(dict(mode=step["mode"], draft_slots=widths, fixed_expected_tokens=11,
                         smallest=min(ratios),mean=sum(ratios)/len(ratios),largest=max(ratios)))

    # Known labels cover reached successes, observed first rejection, stopping and full-accept.
    # Never attribute a failure to a censored tail.
    cal = ConditionalAcceptance()
    proxy = [.68] * DRAFT_SLOTS
    tests = [(2, 2, False), (7, 1, True), (4, 0, False), (7, 7, False)]
    for verified, accepted, rejet in tests:
        cal.observe(7, "greedy", proxy, verified, accepted, rejet)
    expected = [[3,3], [2,3]] + [[1,1]]*5
    assert [cal.count[cal.key(7, "greedy", j, .68)] for j in range(7)] == expected
    assert cal.conditional(7, "sampled", proxy, min_reached=1) is None
    assert cal.conditional(4, "greedy", proxy[:4], min_reached=1) is None
    assert cal.conditional(7, "greedy", proxy) is None
    assert all(0 < x < 1 for x in cal.conditional(7, "greedy", proxy, min_reached=1))
    result = dict(
        scope="Measured conditional logits of the genuine checkpoint selector on synthetic activations and unary top-k candidates. Hypothetical verifier assignments, equal per-row verification cost. These are identifiability bounds on tokens per round.",
        target_accuracy_labels_observed=0,observed_selector_scores=BATCH*DRAFT_SLOTS*len(case),
        actual_greedy_selected_q_values=[1.0],observed_greedy_raw_range=[min(greedy_raw),max(greedy_raw)],
        target_acceptance_multiset=[0,1,3,7],actual_joint_row_budget=BUDGET+BATCH,
        fixed_draft_slots=[3]*BATCH,cases=case,online_label_tests=dict(
            accepted_and_reached_counts=expected,early_stop_is_censored=True,
            unseen_width_sampling_regime_or_sparse_slot_uses_original_policy=True),
        pre_draft_limit="These logits require a completed draft forward. A shorter bidirectional block changes the distribution at early slots. No width-specific Muse traces or checkpoint head for pre-draft prediction were available.")
    (ROOT/"adaptive_evidence.json").write_text(json.dumps(result,indent=2))
    print(len(case),result["observed_greedy_raw_range"],min(x["smallest"] for x in case),max(x["largest"] for x in case),[round(x["mean"],3) for x in case])


if __name__ == "__main__":
    main()
