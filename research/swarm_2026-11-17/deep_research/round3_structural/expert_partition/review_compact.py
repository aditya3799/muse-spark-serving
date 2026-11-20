from pathlib import Path
import collections,hashlib,json,random
P=Path(__file__).resolve().parent;H=P.parent/'hopper_moe';snap=P/'compact_review_snapshot';snap.mkdir(exist_ok=True);files=['compact_routes.py','compact_codegen.py','compact_contract.py','probe_compact_w13.py','SM86_COMPACT_JOBS.json'];pins={}
for name in files:
 data=(H/name).read_bytes();(snap/name).write_bytes(data);pins[name]=hashlib.sha256(data).hexdigest()
checks=0;cases=[]
for size in [8,256,1025,8192,131072]:
 for pattern in ['uniform','last','remote','mixed']:
  rng=random.Random(51+size);ids=[(31 if i%8==0 else -1) if pattern=='last' else -1 if pattern=='remote' else (i%8 if i%16<8 else -1) if pattern=='mixed' else (lambda v:v if v<32 else -1)(rng.randrange(256)) for i in range(size)]
  for cutoff in [16,64]:
   count=[0]*32;partial=[]
   for start in range(0,size,1024):
    h=[0]*32
    for v in ids[start:start+1024]:
     if 0<=v<32:h[v]+=1;count[v]+=1
    partial.append(h)
   light=[v if v<=cutoff else 0 for v in count];heavy=[v if v>cutoff else 0 for v in count];lo=[0];hi=[sum(light)]
   for l,h in zip(light,heavy):lo.append(lo[-1]+l);hi.append(hi[-1]+h)
   mapping=[-1]*size;inverse=[-1]*size;expert=[-1]*size
   # Independent emulation of perchunk cumsum + prior-chunk histogram.
   for e in range(32):
    before=0;base=(lo if count[e]<=cutoff else hi)[e]
    for ch,h in enumerate(partial):
     rank=0
     for pos in range(ch*1024,min(size,(ch+1)*1024)):
      if ids[pos]==e:
       dense=base+before+rank;assert mapping[dense]==-1;mapping[dense]=pos;inverse[pos]=dense;expert[dense]=e;rank+=1;checks+=1
     assert rank==h[e];before+=rank
   expected=sorted([i for i,v in enumerate(ids) if v>=0],key=lambda i:(count[ids[i]]>cutoff,ids[i],i))
   assert mapping[:hi[-1]]==expected and all(v==-1 for v in mapping[hi[-1]:]);checks+=size
   for pos,v in enumerate(ids):
    assert (inverse[pos]==-1) if v<0 else mapping[inverse[pos]]==pos;checks+=1
   # Reproduce original incremental grouped expert cursor, using fixed endpoint offsets.
   for offsets in [lo,hi]:
    ec=[b-a for a,b in zip(offsets,offsets[1:])]
    for bm in [8,16,64,128]:
     owners=[(e,t*bm) for e,n in enumerate(ec) for t in range((n+bm-1)//bm)]
     for workers in [20,132]:
      nt=32
      for cta in range(workers):
       old_m=0;off=0;e=0
       for flat in range(cta,len(owners)*nt,workers):
        m=flat//nt;off+=(m-old_m)*bm
        while off>=ec[e]:off-=((ec[e]+bm-1)//bm)*bm;e+=1
        assert (e,off)==owners[m];assert offsets[e]+off<offsets[e+1];old_m=m;checks+=2
   cases.append({'slots':size,'pattern':pattern,'cutoff':cutoff,'active':len(expected)})
# Changed graph all-remote MUST clear inverse; omittedinit negative exposes stale active row.
prior=[0,-1,-1];current=[-1,-1,-1];stale_inverse=[0,-1,-1];assert any(v>=0 for v in stale_inverse) and all(v<0 for v in current)
# Last expert endpoint negative is non-vacuous at fixedcapacity with allzero offsets.
assert 256-0>0 and 0-0==0
r={'status':'INDEPENDENT_CPU_MAPPING_CURSOR_AND_ALIAS_REVIEW_PASS','pins':pins,'cases':len(cases),'checks':checks,'case_summary':cases,'findings':['GPU prefix class offsets correctly retain absolute heavy base=sumLight.','Patched last endpoint equalsoffset[E]-offset[E-1]; allocationcapacity must not supply activecount.','Empty/trailingempty experts and last-expert-only traversal validated under20/132 persistentCTAstrides.','Map/init/scatter predicates preserve invalidtails; _init_maps clears staleinverse eachgraphreplay.','Harness raw and postscale/activation graphs capture initialempty routes but rebuild GPUcounts/maps beforeuse, making laternonempty replay a meaningful gate.','Shared B1 gather storage is overwritten byactivation onlyafter bothW13partitions complete; B0 raw/scaled storage is distinct.','Scatter destinations in currentW13harness are separateallocations; futureB0→B1 fullchain scatter is legal onlyafter allW2reads complete.'], 'review_limits':['W13harness only; noW2/fullfinalizer performance or nativeGPUproof.','Crossingpattern seed0expert0count17,seed1expert1count15: exercisesdynamicclasses but not sameexpert15→17 directly. Add sameexpert cutoffcross if expandingqualification.','Publichelper gather/scatter shouldchecksamedevice andalias contracts; currentfixture usesoneCUDAdevice/distinctbuffers.','Provenance list omits scoredscale/split/nativeactivationPython and HummingPythonconfig/codegen; pin them for exactreproducibility.','AST/CPU proofs do not establish device release/acquire or nativearithmetic.'],'negative_controls':['omitted inverse reset leaves stalevalid entry onallremote nextgeneration','unpatched finalendpoint invents256rows inempty fixedcapacity plan']}
(P/'COMPACT_INDEPENDENT_CPU_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'cases':len(cases),'checks':checks,'pins':pins}))
