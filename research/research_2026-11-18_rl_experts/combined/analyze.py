"""Create per-scope receipts from the raw shared three-arm pair measurements."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

here=Path(__file__).resolve().parent
scorer=here.parents[1]/"research_2026-11-18_rl/score_trials.py"
manifest=json.loads((here/"composition.json").read_text())
fused=manifest["inputs"]["candidate.diff:research_2026-11-18_rl_prefill"]
summary={}
for comparator in ("parent","ballot"):
    for scope in ("moe","ffn"):
        stem=f"{scope}_{comparator}"
        dirs={}
        rows={}
        for seed in ("pilot","holdout"):
            source=json.loads((here/(seed+".json")).read_text())
            assert source["complete"] and source["correct"] and source["active"]
            shapes=source["bench"]["shapes"] if comparator=="parent" else source["incremental"]
            filtered={k:v for k,v in shapes.items() if k.startswith(scope+"_") and
                (comparator=="parent" or k.split("_")[1] in ("M768","M3072"))}
            raw={k:source[k] for k in ("agent","candidate","candidate_sha256","baseline_sha256","seed","unit","active","correct","complete")}
            raw.update(scope=f"{scope} against {comparator}",bench={"instances":1,"shapes":filtered})
            if comparator=="ballot":
                raw["comparison_reference_sha256"]=fused
                raw["baseline_sha256"]=fused
            dest=here/f"{stem}_{seed}.json";dest.write_text(json.dumps(raw,indent=2)+"\n");dirs[seed]=dest
            rows[seed]={}
            for name,r in filtered.items():
                a,b,c=(r["times"][k] for k in ("baseline","candidate","control"))
                rows[seed][name]=dict(reference_us=statistics.median(a),combined_us=statistics.median(b),
                    control_us=statistics.median(c),ratio=statistics.median(a)/statistics.median(b))
        out=here/f"{stem}_reward.json"
        # The plugin verifier pins the parent SHA. For the second comparison,
        # call its same paired estimator directly with the actual comparator hash.
        if comparator=="parent":
            cmd=[sys.executable,str(scorer),str(dirs["pilot"]),"--holdout",str(dirs["holdout"]),"--output",str(out)]
            subprocess.run(cmd,check=True)
        else:
            import importlib.util
            spec=importlib.util.spec_from_file_location("score_trials",scorer)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            plugin=importlib.util.spec_from_file_location("kar_stats",mod.PLUGIN)
            stats=importlib.util.module_from_spec(plugin);plugin.loader.exec_module(stats)
            runs=[]
            for path in dirs.values():
                data=json.loads(path.read_text())
                assert data["baseline_sha256"]==fused and data["active"] and data["correct"]
                bench=data["bench"];mod.check_bench(bench)
                cmp=stats.compare(bench,["candidate"],["baseline"])
                noise=stats.compare(bench,["control"],["baseline"])
                verdict,why=stats.decide(cmp,noise)
                runs.append(dict(gain=1+cmp["effect"],ci=[1+x for x in cmp["ci"]],
                                 decision=verdict,reason=why,shapes=cmp["shapes"],noise=noise))
            out.write_text(json.dumps({"reference_sha256":fused,"runs":runs},indent=2)+"\n")
        summary[stem]={"rows":rows,"reward":out.name}
(here/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


