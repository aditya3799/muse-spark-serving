"""Run both independently qualified attention fixtures on the composed source."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QKV = ROOT / "research_2026-11-18_rl_prefill/decode_qkv"
ROWS = ROOT / "research_2026-11-18_rl_prefill/decode_pages/global_rows"
SELECT = ROOT / "research_2026-11-18_window_repair/decode_selector"
SHA = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest = json.loads((HERE / "direct_three_manifest.json").read_text())
    patch = HERE / "direct_three_candidate.diff"
    assert SHA(patch) == manifest["sha256"]
    proof = {"bundle_sha256": SHA(patch), "source_identity": {}, "components": []}
    # H.parents[1] must resolve to the repository for the existing checks.
    with tempfile.TemporaryDirectory(dir=HERE.parent) as tmp:
        h = Path(tmp)
        env = dict(os.environ, GIT_INDEX_FILE=str(h / "index"), CUDA_VISIBLE_DEVICES="-1")
        def git(*args):
            return subprocess.check_output(["git", "-C", str(ROOT/"sglang"), *args], env=env)
        def stage(p):
            git("read-tree", manifest["base"])
            git("apply", "--cached", str(p))
        def run_test(folder, file, output):
            shutil.copyfile(folder/file, h/file)
            subprocess.run([sys.executable, str(h/file)], env=env, capture_output=True, check=True)
            return json.loads((h/output).read_text())
        stage(patch)
        entries = {}
        for item in manifest["components"]:
            for path in item["paths"]:
                if path not in entries:
                    entries[path] = git("show", ":"+path)
                # Two distinct dflash.py files do not belong to the attention fixtures.
                if path in manifest["components"][0]["paths"] or path in manifest["components"][2]["paths"]:
                    (h/Path(path).name).write_bytes(entries[path])
        for component in manifest["components"]:
            stage(ROOT/component["patch"])
            for path in component["paths"]:
                if path.endswith("/flashattention_backend.py"):
                    continue
                assert entries[path] == git("show", ":"+path)
                proof["source_identity"][path] = hashlib.sha256(entries[path]).hexdigest()
        shutil.copyfile(patch, h/"candidate.diff")
        stage(ROOT/"release_2026-11-18/candidate.diff")
        (h/"parent_muse_spark.py").write_bytes(git("show", ":python/sglang/srt/models/muse_spark.py"))
        q = run_test(QKV,"dispatch.py","dispatch.json")
        w = run_test(QKV,"preflight.py","eligibility.json")
        d = run_test(ROWS,"preflight.py","dispatch.json")
        assert q["n"] == 60 and q["cumulative_apply_check"]
        assert w["candidate_sha256"] == d["sha"] == SHA(patch)
        assert w["cast_table_all_65536_cpu"] == "matched" and w["scored_verify_graph_width_8"]
        assert d["cases"] == 48 and d["same_ids_storage_across_replay"]
        assert d["shared_read_ends"]["True_verify"] == d["shared_read_ends"]["True_decode"] == "post"
        proof.update(qkv_dispatch_cases=q["n"], qkv_preflight=w, rows_dispatch=d)
    for fn in ["real_pilot.json", "real_holdout.json"]:
        r=json.loads((ROWS/fn).read_text())
        assert r["correct"] and r["complete"] and len(r["bench"]["shapes"])==15
        assert r["stream_ownership"]=={"pre_event_preserved":False,"post_event_preserved":True}
        proof["components"].append({"file":str((ROWS/fn).relative_to(ROOT)),"sha256":SHA(ROWS/fn),
                                    "seed":r["seed"],"memory":r["memory"],"stream_ownership":r["stream_ownership"]})
    lifecycle=json.loads((SELECT/"lifecycle_cpu.json").read_text())
    assert all(lifecycle[k] for k in ("changed_every_step","distinct_rng_successive_calls","static_addresses","all_tokens_q_ids_equal"))
    proof["sampler"]={"worker_and_model_byte_identity":True,"lifecycle_steps":lifecycle["steps"],
                      "lifecycle_proof_sha256":SHA(SELECT/"lifecycle_cpu.json"),"scope":lifecycle["scope"]}
    proof["scope"]="Composed-source CPU dispatch, alias ownership, exact digest matching to independently GPU-qualified kernels; no joint hardware replay or throughput measurement."
    (HERE/"direct_three_verification.json").write_text(json.dumps(proof,indent=2))
    print(proof["bundle_sha256"], q["n"], d["cases"], "ok")
if __name__=="__main__":
    main()

