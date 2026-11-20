from pathlib import Path
import json,random,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parents[3];V=ROOT/'swarm_2026-11-16/moe/humming_vendor/humming'
def plan(ids,experts,cutoff):
 counts=[ids.count(e) for e in range(experts)];light=[c if c<=cutoff else 0 for c in counts];heavy=[c if c>cutoff else 0 for c in counts]
 lo=[0];hi=[sum(light)]
 for a,b in zip(light,heavy):lo.append(lo[-1]+a);hi.append(hi[-1]+b)
 slots=[]
 for group in (light,heavy):
  for e,c in enumerate(group):
   if c:slots.extend(i for i,v in enumerate(ids) if v==e)
 inverse=[-1]*len(ids)
 for pos,slot in enumerate(slots):inverse[slot]=pos
 return {'counts':counts,'light_offsets':lo,'heavy_offsets':hi,'compact_to_slot':slots,'slot_to_compact':inverse,'total':len(slots)}
def audit():
 rows=[]
 for m in (1,8,32,128,256,512,1024,16384):
  for pattern in ('uniform','onehot','eighthot','remote','last','duplicate'):
   rng=random.Random(27+m);ids=[]
   for token in range(m):
    row=rng.sample(range(256),8) if pattern=='uniform' else ([0]+rng.sample(range(32,256),7) if pattern=='onehot' else rng.sample(range(8),8) if pattern=='eighthot' else rng.sample(range(32,256),8) if pattern=='remote' else [31]+rng.sample(range(32,256),7) if pattern=='last' else [0]*8)
    ids.extend(v if v<32 else -1 for v in row)
   for cutoff in (8,16,24,64):
    q=plan(ids,32,cutoff);assert sorted(q['compact_to_slot'])==[i for i,v in enumerate(ids) if v>=0];assert len(set(q['compact_to_slot']))==q['total'];assert all(q['compact_to_slot'][pos]==slot for slot,pos in enumerate(q['slot_to_compact']) if pos>=0)
    for group,key in ((0,'light_offsets'),(1,'heavy_offsets')):
     off=q[key];assert all(a<=b for a,b in zip(off,off[1:]));assert off[-1]<=len(ids)
     for e in range(32):assert all(ids[q['compact_to_slot'][pos]]==e for pos in range(off[e],off[e+1]))
    rows.append({'m':m,'pattern':pattern,'cutoff':cutoff,'active':q['total'],'legacy_last_expert_overcount_light':len(ids)-q['light_offsets'][-1],'legacy_last_expert_overcount_heavy':len(ids)-q['heavy_offsets'][-1]})
 return rows
if __name__=='__main__':
 rows=audit();sources=['include/humming/scheduler.cuh','include/humming/kernel/humming_ws.cuh','include/humming/epilogue/gmem_writer.cuh','csrc/launcher/launcher.cpp','csrc/launcher/tensor.h','transform.py','tune/sm90.py'];r={'status':'CPU_LAYOUT_AND_LIFETIME_DESIGN','checks':len(rows),'cases':rows,'source_hashes':{s:hashlib.sha256((V/s).read_bytes()).hexdigest() for s in sources},'design':{'layout':'Single compact array ordered light/heavy class, then expert, then original slot; GPU offsets E+1 perclass, inverse map preserves slot0..topk-1 finalizer order.','canonical_weights':'Same LayerConfig drives repack. GEMM type changes ComputeConfig and scheduler only; no second weight copy or new quantization.','buffer_lifetime':'B0 capacity S*max(2I,H), B1 S*max(I,H): gatherX toB1; allW13 toB0; originalBF16scale+SiLU toB1; allW2 toB0; scatterdown toB1; unchanged finalizer.','extra_bytes':'AtH=I=4096 no additional largebuffer versus baseline cache13+cache2. General extra=S*max(0,H-I)*2 bytes; maps3*S*4 plus count/offset work.','scheduler_fix':'Last expert count must use offsets[E]-offsets[E-1], not allocation shape_m-offsets[E-1], so GPUvalidtotal can differ from fixedcapacity.','primary_experiment':'SM86 same native MMA/Kpartition config indexed vs grouped_contiguous, rawW13 first; legal routing and allremote/last expert/duplicates, changing graph routes, sentinel. Then fullchain raw/scaled/activation/W2/finalizer and overhead-inclusive matched timing.','Hopper':'Native grouped TMA A/B/scales/C+warp-specialization with pinned grouped epilogue tensor-map clipping. No streamK or FP16acc. Offline compilation only until H200exists.','numerical_scope':'First isolate samegeometry layout effects. Any sameprecision shape/scheduler floating reduction difference is a diagnostic, not forbiddenprecision nor serving-validation.'},'primary_urls':['https://raw.githubusercontent.com/vllm-project/humming/main/humming/include/humming/scheduler.cuh','https://raw.githubusercontent.com/novitalabs/chord/main/docs/design.md','https://vllm-project.github.io/2026/10/19/humming.html']};(P/'COMPACT_GROUPED_DESIGN.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print('CPUcases',len(rows))
