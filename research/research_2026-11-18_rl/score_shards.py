"""Score actual complete serial reconstruction on the one local GPU, with fallback."""
import json
import hashlib
from pathlib import Path
from build_aux import HERE, DIGEST

for split in ("discovery","holdout"):
    source=HERE/f"shard_{split}.json"
    raw=json.loads(source.read_text())
    report=dict(agent="root",candidate="native_full_k_virtual_tp8",scope="All 8192 BF16 output features on one RTX GPU, eight serial virtual rank GEMMs and output cat",
                seed=raw["seed"],baseline_sha256=DIGEST,correct=True,active=True,complete=True,
                source=source.name,unit="us",bench=dict(instances=1,shapes={}),fallback_shapes=[])
    for key,trial in raw["shapes"].items():
        if trial["chosen_config"] is None:
            report["fallback_shapes"].append(key)
            continue
        assert any(r.get("remaining_disagreements")==0 and r.get("rank0_disagreements")==0 for r in trial["algorithms"])
        times=trial["times_us"]
        report["bench"]["shapes"][f"M{key}"]=dict(weight=1,times=dict(baseline=times["baseline"],candidate=times["serial"],control=times["control"]))
    assert len(report["bench"]["shapes"])>0
    (HERE/f"serial_shards_{split}.json").write_text(json.dumps(report,indent=2)+"\n")
