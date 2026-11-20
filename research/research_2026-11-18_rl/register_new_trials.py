"""Translate the new full-width local experiments into the signed comparison protocol."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from score_trials import BASE

HERE=Path(__file__).resolve().parent
def record(source,agent,candidate,scope,correct,shapes=None):
    with source.open(encoding="utf-8") as file: raw=json.load(file)
    if raw["baseline"]!=BASE or not raw["complete"]:raise ValueError(source)
    result=dict(agent=agent,candidate=candidate,scope=scope,correct=correct,active=True,complete=True,
                baseline_sha256=BASE,seed=raw["seed"],variant_identity=candidate,
                original_receipt=source.name,original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                bench=dict(instances=1,shapes=shapes or {}))
    flat=source.with_name(source.stem+"_paired.json")
    target=source.with_name(source.stem+"_reward.json")
    flat.write_text(json.dumps(result,indent=2))
    subprocess.run([sys.executable,"-B",str(HERE/"score_trials.py"),str(flat),"--output",str(target)],check=True)


def feature(name):
    p=HERE/"feature_shard"/name
    if not p.exists():return
    d=json.loads(p.read_bytes())
    shapes={key:dict(weight=1,times=row["ms"]) for key,row in d["rows"].items()}
    exact=all(s["raw_mismatches"]==s["final_bit_mismatches"]==0 and s["finite"] and s["changing"] for r in d["rows"].values() for s in r["parity"])
    record(p,"feature-sharding","paired-feature-shards-"+("matched-tiles" if d.get("preserve_tiles") else "native-tiles"),
           "Full-width two-rank-serial Humming FFN on SM86; local concatenation and normalization; no NCCL transport",exact,shapes)


def placement(name):
    p=HERE/"placement"/name
    if not p.exists():return
    d=json.loads(p.read_bytes())
    rows=d["rows"]; repair=all(r.get("returned") is not None for r in rows.values())
    exact=all((r["returned"]["sm86_original_order_exact"] if repair else r["rel_l2"]==0.) and r["shuffled_expert_raw_mismatches"]==0 and r["control_bits_equal"] and r["changed_all"] and r["finite"] and r["all_ranks"] for r in rows.values())
    shapes={key:dict(weight=1,times=r["per_round_max"]) for key,r in rows.items()}
    record(p,"expert-placement","same-run-slot-exchange-"+("restored-owners" if repair else "new-owner-finalize"),
           "Peak of eight sequentially replayed SM86 rank-local experts and finalizer; migration and cross-rank transfer excluded",exact,shapes)


if __name__=="__main__":
    for filename in ("fullwidth_pilot.json","matched_pilot.json"):feature(filename)
    for filename in ("allrank_pilot.json","allrank_repaired_real.json"):placement(filename)
