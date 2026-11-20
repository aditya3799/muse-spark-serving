"""Compare maximum-rank structural work; deliberately makes no speed prediction."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tile_study import choose_block, manifest

root = Path(__file__).resolve().parent
out = root / "local_structure"
groups = defaultdict(list)
with (out / "routing_geometry.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["ep"]) == 8:
            groups[
                int(row["tokens"]), row["pattern"], int(row["seed"]), int(row["block"])
            ].append(row)
comparisons = []
for tokens in (8, 32, 128, 256):
    for pattern in ("uniform", "hot_rank", "hot_expert", "repeated"):
        values = {p: defaultdict(list) for p in ("baseline", "ep_density")}
        for policy in values:
            block = choose_block(tokens, 8, 32, 256, policy)
            for seed in range(4):
                ranks = groups[tokens, pattern, seed, block]
                for field in ("active_padded_rows", "active_blocks"):
                    values[policy][field].append(max(int(row[field]) for row in ranks))
        comparisons.append(
            dict(
                tokens=tokens,
                pattern=pattern,
                **{
                    p: {key: statistics.median(v) for key, v in fields.items()}
                    for p, fields in values.items()
                },
            )
        )
(out / "structural_comparison.json").write_text(
    json.dumps(
        {
            "scope": "median of per-seed maximum-rank counts; these are not timings",
            "comparisons": comparisons,
        },
        indent=2,
    ),
    encoding="utf-8",
)
manifest(out)
print(json.dumps([r for r in comparisons if r["tokens"] == 128], indent=2))
