from pathlib import Path
import ast,difflib,hashlib,json,subprocess
P=Path(__file__).resolve().parent;A=P.parent;ROOT=A.parents[2];GIT=ROOT/'sglang';SCORED=ROOT/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused'
head=subprocess.check_output(['git','-C',str(GIT),'rev-parse','HEAD'],text=True).strip();assert head=='297e54fd105e974cdf401e5b4d20253a22c16f90'
scored={p.relative_to(SCORED/'candidate_sources').as_posix():p for p in (SCORED/'candidate_sources').rglob('*.py')}
review=P/'package_review';review.mkdir(exist_ok=True);records=[]
for variant in ['hot_direct','hot_fused']:
    overlay={p.relative_to(P/(variant+'_sources')).as_posix():p for p in (P/(variant+'_sources')).rglob('*') if p.is_file() and p.suffix in ['.py','.cuh']}
    merged=dict(scored);merged.update(overlay);parts=[];expected={};originals={};overlap=sorted(set(scored)&set(overlay));guarded=[]
    for rel,path in sorted(merged.items()):
        assert rel.startswith('python/sglang/') and not rel.startswith('python/sglang/srt/model_loader/')
        assert path.suffix in ['.py','.cuh','.json']
        candidate=path.read_text()
        if path.suffix=='.py':ast.parse(candidate)
        result=subprocess.run(['git','-C',str(GIT),'show',head+':'+rel],capture_output=True,text=True)
        base=result.stdout if result.returncode==0 else ''
        originals[rel]=base;expected[rel]=candidate
        parts.extend(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+rel if base else '/dev/null',tofile='b/'+rel))
        if rel in overlay and rel not in scored and base and (GIT/rel).exists():
            guarded.append({'path':rel,'pinned_matches_working_source':base==(GIT/rel).read_text()})
    patch=''.join(parts);added=[line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++')]
    assert len(patch.encode())<=400000 and max(map(len,added))<=400
    patchfile=review/(variant+'_scored_composition.diff');patchfile.write_text(patch)
    tree=review/(variant+'_roundtrip');tree.mkdir(exist_ok=True)
    for rel,base in originals.items():
        if base:
            dst=tree/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(base)
    check=subprocess.run(['git','apply','--check',str(patchfile)],cwd=tree,capture_output=True,text=True)
    applied=subprocess.run(['git','apply',str(patchfile)],cwd=tree,capture_output=True,text=True) if check.returncode==0 else None
    exact=applied is not None and applied.returncode==0 and all((tree/rel).read_text()==text for rel,text in expected.items())
    records.append({'variant':variant,'files':len(merged),'bytes':len(patch.encode()),'max_added_line':max(map(len,added)),'overlap_with_scored':overlap,'pin_source_checks':guarded,'git_apply_check':check.returncode,'git_apply_exit':applied.returncode if applied else None,'roundtrip_exact':exact,'errors':check.stderr+(applied.stderr if applied else ''),'sha256':hashlib.sha256(patch.encode()).hexdigest()})
    assert exact,records[-1]
report={'status':'ISOLATED_SCORED_COMPOSITION_ROUNDTRIP_PASS','pinned_head':head,'scored_patch_sha256':hashlib.sha256((SCORED/'candidate.diff').read_bytes()).hexdigest(),'variants':records,'scope':'Local draft package validation only; never submitted. Only overlap with ten-file scored source set is muse_spark.py, whose scored body is preserved beneath explicit auxiliary changes. No active checkout mutations.'}
(P/'SCORED_COMPOSITION_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
