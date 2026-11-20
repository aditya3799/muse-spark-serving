import collections,hashlib,json,pathlib,random
p=pathlib.Path(__file__).resolve().parent;other=p.parent/'hopper_moe';names=['probe_legal_selective_m2.py','selective_routes.py','LEGAL_M32_JOBS.json','virtual_k_codegen.py','ACTUAL_SCORED_POLICY_AND_LEGAL_ROUTES.json'];snap=p/'hopper_selective_review_snapshot';snap.mkdir(exist_ok=True);hashes={}
for name in names:
 data=(other/name).read_bytes();(snap/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
jobs=json.loads((snap/'LEGAL_M32_JOBS.json').read_text());audit=json.loads((snap/'ACTUAL_SCORED_POLICY_AND_LEGAL_ROUTES.json').read_text());actual=next(row for row in audit['rows'] if row['m']==32);field_checks=[]
for arm in ('light','heavy'):
 cfg=next(j['configuration'] for j in jobs['jobs'] if j['label']==arm)
 for field,value in actual[arm].items():
  if field=='num_sms':continue
  assert cfg[field]==value,(arm,field);field_checks.append((arm,field))
checks=0
for c in range(257):
 light=c<=16;eligible=c>16 and (c+31)//32<(c+15)//16;heavy=not light and not eligible
 assert int(light)+int(eligible)+int(heavy)==1
 assert not heavy
 checks+=2
report={'status':'CPU_REVIEW_NO_BLOCKING_REFERENCE_OR_CAPTURE_ISSUE','snapshot_sha256':hashes,'reference_field_checks':len(field_checks),'partition_predicate_checks':checks,'findings':[
 'Original light/heavy audited config fields match; num_sms20 is absent from explicit kernel config and runtime uses local SM count, which is20 for this fixture.',
 'render_virtual_k enabledFalse returns source unchanged, preserving baseline kernel expression.',
 'For cutoff16, heavyBM16, M2BM32, every count>16 is eligible. Original-heavy third partition is provably empty even for duplicate routes; its launch overhead is included but cannot receive nonempty coverage.',
 'GPU histogram excludes invalid IDs into dedicated bucket; light/eligible/remaining-heavy are exhaustive and disjoint with preserved canonical slot IDs.',
 'Histogram, classification, alignment and all GEMMs occur inside capture; changed IDs and scales retain pointer identity and execute updated routing.',
 'Raw output, scaled output and stock activation compare finite and bitwise. Sentinel fills precede each replay and inactive outputs are verified unchanged.',
 'Stage capture rewrites active W13 output before in-place BF16 scaling each replay, avoiding repeated scale accumulation.',
 'All five route fixtures assert globally unique top8; all-remote arithmetic comparison is empty but full inactive sentinel check is nonvacuous.',
 'Timing includes one histogram, all alignments and GEMMs plus scale/SiLU on both sides, including candidate empty third launch. A/A,B/B and half-block drift controls are explicit.',
 'Source unchanged audit is checked at completion; unsuccessful partial reports must remain incomplete unless final status completed.'
 ],'limits':['W13 stage only; no W2/finalizer/full-helper integration','Ordinary canonical W13 representation only; custom M2 interleaved weight+scale/activation interaction remains untested','SM86 native M32 only; no H200 inference','Hostile positive out-of-range IDs are not part of legal route fixtures; gate remaps only actual global routes to local or-1','Three-partition wrapper is deliberately narrow research code, not a generic serving selector'],'nonblocking_note':'CLI error text says M1..256 although actual accepted tokens list is exactly[32]; guard itself is correct.'}
(p/'HOPPER_SELECTIVE_CPU_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'hashes':hashes,'field_checks':len(field_checks),'partition_checks':checks}))
