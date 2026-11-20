import re,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
H=ROOT.parent/'hopper_moe'/'offline_compact_01'
def parse(path):
    ins={}; labels={}; pending=[]
    for line in path.read_text().splitlines():
        m=re.match(r'(\.L_x_\d+):',line)
        if m: pending.append(m.group(1))
        m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
        if not m: continue
        pc=int(m.group(1),16); ins[pc]=m.group(2)
        for label in pending: labels[label]=pc
        pending=[]
    edges={}
    pcs=sorted(ins)
    for i,pc in enumerate(pcs):
        s=ins[pc]; nex=pcs[i+1:i+2]; lab=re.search(r'`\((\.L_x_\d+)\)',s)
        targets=[labels[lab.group(1)]] if lab else []
        # Conservative reconvergence/collective approximation: both fallthrough and explicit targets.
        if re.search(r'\bBRA(?:\.|\s)',s):
            edges[pc]=targets+(nex if s.startswith('@') or 'BRA.DIV' in s else [])
        elif re.search(r'\bEXIT\b',s): edges[pc]=nex if s.startswith('@') else []
        else: edges[pc]=nex+targets
    return ins,edges,labels

def reachable(edges,cuts):
    todo=[0]; seen=set()
    while todo:
        x=todo.pop()
        if x in seen: continue
        seen.add(x); todo.extend(cuts.get(x,edges[x]))
    return seen
p=H/'w2_m32_grouped_tma_ws'/'kernel.sass'; ins,edges,labels=parse(p)
assert ins[0x10]=='S2R R0, SR_TID.X'
# R0 is immutable after entry, including vector destinations (none begins below R0).
writes=[hex(pc) for pc,s in ins.items() if re.match(r'(?:@!?\w+\s+)?[A-Z0-9_.]+\s+R0(?:\s|,)',s)]
assert writes==['0x10'],writes
assert ins[0xbe0]=='@P0 BRA `(.L_x_11)'
assert 'PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x0'==ins[0x7b0]
assert ins[0x850]=='@!P3 BRA `(.L_x_7)'
assert 'R0, 0x200, PT' in ins[0x7f0]
assert 'R0, 0x1f, !P0' in ins[0x10f0]
assert ins[0x1120]=='@P0 BRA `(.L_x_20)'
local=[pc for pc,s in ins.items() if re.search(r'\b(?:STL|LDL)(?:\.|\s)',s)]
assert local==[0x1190,0x12c0]
# Consumer proof: TID<512 takes 0x850; P0=true from 0x7b0 survives directly to 0xbe0.
# Producer proof: TID>=512 makes (R0>31)||!expert_changed true at 0x10f0.
cons=reachable(edges,{0x850:[labels['.L_x_7']],0xbe0:[labels['.L_x_11']]})
prod=reachable(edges,{0x1120:[labels['.L_x_20']]})
assert not set(local)&cons
assert not set(local)&prod
negative=reachable(edges,{})
assert set(local)<=negative
q=H/'w13_m768_grouped_tma_ws'/'kernel.sass'; bi,be,bl=parse(q)
assert 'STL [R1+0xe8], R3'==bi[0x700]
assert 'STL [R1+0x14], R0'==bi[0x760]
assert bi[0x8d0]=='@!P0 BRA `(.L_x_7)'
assert 'R8, 0x100, PT' in bi[0x670]
breach=reachable(be,{})
bloc=[pc for pc,s in bi.items() if re.search(r'\b(?:STL|LDL)(?:\.|\s)',s)]
consumer_entry=bl['.L_x_11']
# Entry to consumer code has immediate local reloads before its first MMA.
consumer_local=[pc for pc in bloc if consumer_entry<=pc<0x4e70]
assert 0x700 in breach and 0x760 in breach and consumer_local
report={'scope':'CPU decoded-SASS reachability with explicit source-derived thread predicates. No GPU, no dynamic traffic count, not a full ISA proof.',
 'sources':{str(x.relative_to(ROOT.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,q]},
 'w2_m32':{'R0_writes':writes,'static_local_instruction_addresses':[hex(x) for x in local], 'consumer_cut':{'tid':'0..511','0850':'takes skip-producer-init','0be0':'P0=true; jumps consumer'},'producer_cut':{'tid':'>=512','1120':'P0=(tid>31)||!expert_changed=true; skips descriptor'},'consumer_reachable_nodes':len(cons),'producer_overapprox_nodes':len(prod),'local_reachable_consumer':False,'local_reachable_producer':False,'negative_without_thread_cuts_reaches_both':True,'qualification':'Compiler-reported 8-byte frame/spill receipt unchanged. These two local instructions cannot execute under admitted thread split; do not call binary zero-spill.'},
 'w13_bm176':{'static_local_instruction_count':len(bloc),'unconditional_pre_split_stores':['0x700','0x760','0x8c0'],'consumer_entry':hex(consumer_entry),'consumer_local_before_first_mma':[hex(x) for x in consumer_local],'qualification':'Consumer-reachable local traffic exists; unlike W2, this cannot be dismissed as a producer-only dead block. No runtime severity inferred.'},
 'limitations':['Unknown data-dependent branches retain both paths. Explicit BSSY/collective targets are included conservatively. Predicate simplifications are independently source/SASS-derived; no general SASS emulator.','Ptxas allocation and frame/spill summaries are not modified.','No conclusion about native runtime correctness or speed.']}
(ROOT/'COMPACT_SPILL_INDEPENDENT_CFG_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
