"""Routing-only coverage across all ranks; no timings or serving predictions."""
import json
import statistics
from collections import Counter
from pathlib import Path
from tile_study import make_routes

rows = []
for tokens in (127, 128, 129, 255, 256, 257, 511, 512, 513, 1024):
    maxima = []
    for seed in range(200):
        counts = Counter(v for row in make_routes(tokens, 8, 256, 8, "uniform", seed) for v in row)
        maxima.append([max((counts[e] for e in range(rank*32, (rank+1)*32)), default=0) for rank in range(8)])
    rows.append(dict(tokens=tokens, seeds=200, median_global_max_count=statistics.median(max(r) for r in maxima),
                     all_ranks_fit_16_fraction=sum(max(r)<=16 for r in maxima)/200,
                     all_ranks_fit_32_fraction=sum(max(r)<=32 for r in maxima)/200))
result = dict(scope="Synthetic uniform-route counts across eight expert partitions; not simultaneous GPU performance or a real model routing trace", rows=rows)
Path(__file__).with_name("cross_rank_geometry.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
