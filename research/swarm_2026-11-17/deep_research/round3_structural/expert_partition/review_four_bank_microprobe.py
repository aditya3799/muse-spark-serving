from pathlib import Path
import hashlib,json,struct
P=Path(__file__).resolve().parent;A=P.parent/'aux_speculation';r32=lambda x:struct.unpack('f',struct.pack('f',x))[0];add=lambda a,b:r32(a+b)
rows=[]
for p in ([2**24,1,-2**24,1],[2**24,-2**24,1,1]):
 cross=add(add(p[0],p[2]),add(p[1],p[3]));adjacent=add(add(p[0],p[1]),add(p[2],p[3]));left=add(add(add(p[0],p[1]),p[2]),p[3]);rows.append({'partials':p,'pinned_cross':cross,'wrong_adjacent':adjacent,'wrong_leftfold':left})
assert rows[0]['pinned_cross']==2 and rows[1]['pinned_cross']==1
assert rows[0]['wrong_adjacent']==rows[0]['wrong_leftfold']==1
assert rows[1]['wrong_adjacent']==rows[1]['wrong_leftfold']==2
files=[A/f for f in ['four_bank_probe.cuh','four_bank_probe.py','probe_four_bank.py','FOUR_BANK_REFERENCE_CONFIG.json']]
r={'status':'CPU_SOURCE_NO_BLOCKER_ARITHMETIC_ONLY','sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in files},'independent_float32_engineered_checks':rows,'review':['Pinned shared reducer firstpairs banks0+2 and1+3, then updated0+1, preserving per-lane scalar. Explicit __fadd_rn microkernel matches this tree.','Native comparator BM16/BN128/BK256/WK64 preserves fourphysical64wideKbank sequences, two outputNtiles forN256. It is ordinarySM86MMA, not nativeWGMMA.','Engineered q4=code6/scale1 yields quarteroperand1; BF16x values atK0/64/128/192 are exact. Subsequentzeros cannot hide treeassociation.','Allfourteen changedroute generations have unique frontpackedslots and boundcheckedscatter, distinct outputs, poison and inactive sentinel; lasttwo are known-value counterexamples.','Candidate rawFP32 equality is separatelychecked beforeBF16conversion; pinnedreferenceBF16/postscale mandatory. No source precision/tolerance weakening.','unpack and launch inputdtype/shape/device/currentdevice/byteintervalguards preserved from earliercorrectedmicroprobe.'],'exclusions':'No pipeline, nativeHopper,WGMMA,productionW2 or performance qualification. Separate from currenttwo-bankBM8/24pipeline gate.'}
(P/'FOUR_BANK_MICROPROBE_INDEPENDENT_REVIEW.json').write_text(json.dumps(r,indent=2)+'\n');print(rows)
