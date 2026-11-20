"""CPU scheduler/ownership proof and bounded stage-reuse model, not H200 execution."""
import hashlib
import itertools
import json
import random
from collections import deque
from pathlib import Path
from cluster_contract import align_reference, cost_reference, policy_for_cluster

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
V = ROOT / 'swarm_2026-11-16/moe/humming_vendor/humming'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_routes(routes, experts, bm, cluster, nblocks, groups):
    rows, ids, padded = align_reference(routes, experts, bm, cluster)
    assert padded % (bm * cluster) == 0
    expected = [i for i, e in enumerate(routes) if e >= 0]
    assert sorted(r for r in rows if r < len(routes)) == expected
    assert len(ids) * bm == padded
    assignments = {}
    mn_total = padded // (bm * cluster) * nblocks
    # Exactly the non-stream-K pinned scheduler mapping.
    for group in range(groups):
        for mn in range(group, mn_total, groups):
            mgroup, n = divmod(mn, nblocks)
            same_expert = []
            for rank in range(cluster):
                mb = mgroup * cluster + rank
                expert = ids[mb]
                same_expert.append(expert)
                for r in rows[mb * bm:(mb + 1) * bm]:
                    if r < len(routes):
                        assert routes[r] == expert
                        key = (r, n)
                        assert key not in assignments
                        assignments[key] = expert
            assert len(set(same_expert)) == 1
    assert len(assignments) == len(expected) * nblocks
    # The compact IDs returned by BM*C alignment need explicit repetition.
    compact = ids[::cluster]
    assert [e for e in compact for _ in range(cluster)] == ids
    return cost_reference([routes.count(e) for e in range(experts)], bm, cluster)


def stage_protocol(cluster, stages, generations, cross_release):
    """Adversarial producer/consumer ordering for a multicast ring buffer.
    consumed[r] is next generation that rank r may read. Production may
    prefetch into free slots; overwriting any rank's unread generation is bad.
    Cross-release waits for every rank before reuse; local-only waits for rank 0.
    """
    initial = (0, (0,) * cluster)
    q = deque([(initial, [])])
    visited = {initial}
    while q:
        (produced, consumed), trace = q.popleft()
        if produced < generations:
            release = min(consumed) if cross_release else consumed[0]
            if produced - release < stages:
                action = 'multicast generation %d into slot %d' % (produced, produced % stages)
                if produced >= stages and any(c <= produced - stages for c in consumed):
                    return {'safe_in_model': False, 'states': len(visited),
                            'counterexample': trace + [action],
                            'overwritten_generation': produced - stages,
                            'next_unread_per_rank': list(consumed)}
                nxt = (produced + 1, consumed)
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, trace + [action]))
        for rank in range(cluster):
            if consumed[rank] < produced:
                c = list(consumed)
                action = 'rank %d consumes generation %d' % (rank, c[rank])
                c[rank] += 1
                nxt = (produced, tuple(c))
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, trace + [action]))
    return {'safe_in_model': True, 'states': len(visited)}


def main():
    rng = random.Random(17391)
    checks = 0
    cost_cases = []
    # Exhaustive small distributions retain duplicates and invalid EP routes.
    for size in range(7):
        for routes in itertools.product((-1, 0, 1), repeat=size):
            for bm in (1, 2, 3):
                for cluster in (2, 4):
                    verify_routes(list(routes), 2, bm, cluster, 3, 5)
                    checks += 1
    for bm in (8, 16, 32, 48, 64, 96, 128, 176):
        for cluster in (2, 4):
            for tokens in (1, bm - 1, bm, bm + 1, cluster * bm - 1, cluster * bm, cluster * bm + 1, 1024):
                for pattern in ('uniform', 'hot', 'remote', 'one_each'):
                    experts = 32
                    if pattern == 'uniform':
                        routes = [rng.randrange(-1, experts) for _ in range(tokens)]
                    elif pattern == 'hot':
                        routes = [0 if i % 4 else -1 for i in range(tokens)]
                    elif pattern == 'remote':
                        routes = [-1] * tokens
                    else:
                        routes = list(range(min(tokens, experts))) + [-1] * max(tokens - experts, 0)
                    cost = verify_routes(routes, experts, bm, cluster, 7, 33)
                    checks += 1
                    cost_cases.append(dict(bm=bm, cluster=cluster, routes=tokens, pattern=pattern, **cost))
    protocol = []
    for cluster in (2, 4):
        for stages in (3, 4):
            for cross in (False, True):
                outcome = stage_protocol(cluster, stages, stages + 3, cross)
                assert outcome['safe_in_model'] == cross
                protocol.append(dict(cluster=cluster, stages=stages, cross_release=cross, **outcome))
    original = dict(block_shape=[64, 256, 64], warp_shape=[64, 32, 64], num_stages=4,
                    use_stream_k=False, use_f16_accum=False, use_warp_spec=True,
                    use_tma_b=True, use_mbarrier=True, multi_cast_size_a=1)
    config, reason = policy_for_cluster(original, 90, 2, 132)
    assert config and config['multi_cast_size_b'] == 2
    policy_checks = 1
    for field, value in [('use_warp_spec', False), ('use_stream_k', True), ('use_f16_accum', True),
                         ('use_tma_b', False), ('use_mbarrier', False), ('use_tma_a', True),
                         ('multi_cast_size_a', 2), ('num_stages', 2)]:
        mutated = dict(original, **{field:value})
        assert policy_for_cluster(mutated, 90, 2, 132)[0] is None
        policy_checks += 1
    assert policy_for_cluster(original, 86, 2, 132)[0] is None
    assert policy_for_cluster(original, 90, 4, 131)[0] is None
    policy_checks += 2
    paths = ['include/humming/scheduler.cuh', 'include/humming/kernel/humming.cuh',
             'include/humming/kernel/humming_ws.cuh', 'include/humming/memory/g2s_pipeline.cuh',
             'include/humming/memory/g2s_loader/loader_b.cuh', 'csrc/launcher/launcher.cpp']
    non_ws = (V / paths[1]).read_text()
    ws = (V / paths[2]).read_text()
    assert 'consumer.arrive(' not in non_ws and 'producer.wait_stage(' not in non_ws
    assert 'consumer.arrive(stage_id)' in ws and 'producer.wait_stage(stage_id)' in ws
    report = {'status':'PASS_CPU_CONTRACT_WITH_NON_WS_COUNTEREXAMPLE',
              'scope':'CPU metadata and abstract interleaving only; no CUDA, H200 numerical or timing evidence',
              'routing_cases': checks, 'policy_guards': policy_checks,
              'protocol_cases':protocol, 'cost_cases':cost_cases,
              'source_hashes':{str(p):sha(p) for p in [Path(__file__), P/'cluster_contract.py', P/'cluster_alignment.py']},
              'pinned_humming_hashes':{f:sha(V/f) for f in paths},
              'new_finding':'Matching expert IDs is necessary but insufficient. Non-WS path has no remote-stage release before rank-0 multicast reuse. Research policy fails closed for non-WS.',
              'remaining':'Actual WS correctness at MN transitions, synchronization sanitizers, graph replay, clustered occupancy, spills, padding and whole-chain performance require H200 execution.'}
    (P/'cpu_contract_results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','routing_cases','policy_guards')}))
    print('Protocol explored states:', sum(x['states'] for x in protocol))

if __name__ == '__main__':
    main()
