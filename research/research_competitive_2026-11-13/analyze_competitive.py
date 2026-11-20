"""Reproducible arithmetic and synthetic routing analysis; no GPU speed predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
SERVING = ROOT.parent
HBM_BYTES_S = 4.8e12
WORLD = 8
HIDDEN = 8192
EXPERT_HIDDEN = 4096
INTERMEDIATE = 4096
TOPK = 8
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = 32


def scalar_float(x):
    return float(x)


def main():
    patch = SERVING / 'submission_v4.diff'
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert patch_sha == '92054389830a8e5c07a37eb524810ac83690fbbcf88c60e3ae8a132909bb65f1'
    donor_paths = {
        'moe_v139': '../muse_moe_w4a16/submission_v139_wgs_kperm3.py',
        'attention': '../muse_attention_decode/submission.py',
        'decode_s081': '../muse_decode_layer/submission_s081_keell0renz_890809c.py',
        'decode_active_unmatched_to_s081': '../muse_decode_layer/submission.py',
        'prefill_final_valid': '../muse_prefill_layer/submission_scored_399cbda410c440ee.py',
    }
    donors = {}
    for name, relative in donor_paths.items():
        path = SERVING / relative
        data = path.read_bytes()
        donors[name] = {'path': str(path.resolve()), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()}

    projection_rows = []
    weight_bytes = HIDDEN * EXPERT_HIDDEN * 2
    for m in (8, 32, 128, 256, 1024, 4096, 16384):
        for n, k in ((EXPERT_HIDDEN, HIDDEN), (HIDDEN, EXPERT_HIDDEN)):
            output_bytes = m * n * 2
            projection_rows.append({
                'tokens': m, 'n': n, 'k': k,
                'replicated_weight_bytes_per_rank': weight_bytes,
                'sharded_weight_bytes_per_rank': weight_bytes // WORLD,
                'full_output_bytes': output_bytes,
                'remote_output_bytes_per_rank': output_bytes * (WORLD - 1) // WORLD,
                'gemm_flops_per_rank_replicated': 2 * m * n * k,
                'gemm_flops_per_rank_sharded': 2 * m * n * k // WORLD,
                'ideal_cold_weight_stream_us_full': weight_bytes / HBM_BYTES_S * 1e6,
                'ideal_cold_weight_stream_us_shard': weight_bytes / WORLD / HBM_BYTES_S * 1e6,
                'caveat': 'Ideal byte accounting only; compute, collectives, cache, launches and imbalance excluded.'})

    expert_values = 3 * EXPERT_HIDDEN * INTERMEDIATE
    expert_bytes = expert_values // 2 + expert_values // 16
    moe_rows = []
    for m in (8, 32, 128, 256):
        active = LOCAL_EXPERTS * (1 - (1 - TOPK / GLOBAL_EXPERTS) ** m)
        moe_rows.append({'tokens': m, 'independent_uniform_expected_unique_local_experts': active,
                         'raw_bytes_per_expert': expert_bytes,
                         'ideal_cold_single_read_weight_us': active * expert_bytes / HBM_BYTES_S * 1e6,
                         'caveat': 'Not a bound on real correlated routing. Excludes all activation traffic and computation.'})

    # An explicit repeat probability models within-request route correlation.
    # Draw fresh random parent routes and expert skew per seed. No judge data.
    routing = []
    for m in (8, 32, 128, 256):
        for repeat in (0.0, 0.5, 0.9):
            for sigma in (0.0, 1.0):
                per_tile = {b: [] for b in (8, 16, 32, 48, 64)}
                actual_rows, active_counts = [], []
                for seed in range(100):
                    rng = np.random.default_rng(seed)
                    logits = rng.normal(0, sigma, GLOBAL_EXPERTS)
                    def draw():
                        return np.argpartition(logits + rng.gumbel(size=GLOBAL_EXPERTS), -TOPK)[-TOPK:]
                    routes = []
                    for start in range(0, m, 8):
                        parent = draw()
                        for _ in range(min(8, m - start)):
                            routes.extend(parent if rng.random() < repeat else draw())
                    hist = np.bincount(routes, minlength=GLOBAL_EXPERTS).reshape(WORLD, LOCAL_EXPERTS)
                    rank = hist.sum(axis=1).argmax()
                    local = hist[rank]
                    actual_rows.append(int(local.sum()))
                    active_counts.append(int(np.count_nonzero(local)))
                    for b in per_tile:
                        tiles = int(((local + b - 1) // b).sum())
                        per_tile[b].append((tiles, tiles * b))
                routing.append({'tokens': m, 'request_route_repeat_probability': repeat,
                                'expert_logit_sigma': sigma,
                                'mean_routes_on_busiest_rank': scalar_float(np.mean(actual_rows)),
                                'mean_active_experts_on_busiest_rank': scalar_float(np.mean(active_counts)),
                                'tile_accounting': {
                                    str(b): {'mean_tiles': scalar_float(np.mean(v, axis=0)[0]),
                                             'mean_padded_rows': scalar_float(np.mean(v, axis=0)[1])}
                                    for b, v in per_tile.items()}})

    # Amdahl applies to fixed-work latency, NOT the official frontier score.
    amdahl = [{'untouched_runtime_share': tax, 'optimized_region_speedup': s,
               'fixed_work_latency_speedup': 1 / (tax + (1 - tax) / s)}
              for tax in (0.0, 0.1, 0.2, 0.3) for s in (2.26, 3.0, 4.0)]
    result = {
        'scope': 'Analytical scenarios, not target measurements or frontier-score predictions',
        'frozen_v4_sha256': patch_sha,
        'donors': donors,
        'score_math': {'one_load_3x_other_three_1x_geo': 3 ** .25,
                       'one_load_gain_needed_for_overall_3x_if_others_1x': 3 ** 4},
        'projection_accounting': projection_rows,
        'moe_v139_packed_storage': {
            'raw_one_layout_bytes_per_rank_per_layer': expert_bytes * LOCAL_EXPERTS,
            'persistent_layout_count': 8,
            'packed_layouts_bytes_per_rank_per_layer': 8 * expert_bytes * LOCAL_EXPERTS,
            'excess_over_one_layout_bytes': 7 * expert_bytes * LOCAL_EXPERTS,
            'scope': 'Two prepare states, each base + fat + unit + WGS. Excludes original weights, scratch and conversion peaks.'},
        'moe_weight_accounting': moe_rows,
        'synthetic_correlated_routing': routing,
        'amdahl_fixed_work_scenarios': amdahl,
        'sources': {'h200_spec': 'https://www.nvidia.com/en-us/data-center/h200/',
                    'model_geometry': 'sglang/python/sglang/srt/configs/muse_spark.py'},
    }
    (ROOT / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'donors': donors, 'moe_weight_accounting': moe_rows,
                      'routing_scenarios': len(routing), 'amdahl_fixed_work_scenarios': amdahl}, indent=2))


if __name__ == '__main__':
    main()


