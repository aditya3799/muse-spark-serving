from pathlib import Path
here=Path(__file__).resolve().parent
s=(here/'test_split_ready_protocol.py').read_text(encoding='utf-8-sig')
s=s.replace('wr=[None]*slots;xr=[None]*slots','wr=[None]*slots;xr=[None]*slots\n wp=[0]*slots;xp=[0]*slots;jp=[0]*slots;jlast=[None]*slots\n we=[0]*slots;xe=[0]*slots')
s=s.replace("acks[slot].add(g)","acks[slot].add(g)\n  if len(acks[slot])==groups:\n   phase=we if acks is wa else xe\n   phase[slot]^=1")
s=s.replace('target[seq%slots]=seq','''target[seq%slots]=seq
   phase=wp if which==0 else xp
   phase[seq%slots]^=1
   if wr[seq%slots]==xr[seq%slots]==seq and jlast[seq%slots]!=seq:
    jp[seq%slots]^=1;jlast[seq%slots]=seq''')
s=s.replace("if len(acks[slot])!=groups:continue",'''if seq>=slots:
    phase=we if which==0 else xe
    empty_ready=phase[slot]!=(((seq//slots)&1)^1)
    assert empty_ready==(len(acks[slot])==groups),('empty barrier stale parity',seed,which,seq)
    if not empty_ready:continue
   elif len(acks[slot])!=groups:continue''')
s=s.replace("if wr[slot]!=seq or (not split and xr[slot]!=seq):continue",'''phase=wp if split else jp
   ready=phase[slot]!=((seq//slots)&1)
   if not ready:continue
   assert wr[slot]==seq and (split or xr[slot]==seq),('full barrier stale parity',seed,seq)
   ''')
s=s.replace("if bug!='missing_input_wait' and xr[slot]!=seq:continue",'''if bug!='missing_input_wait':
    ready=xp[slot]!=((seq//slots)&1)
    if not ready:continue''')
s=s.replace("stages=count,readiness_checks=checks", "stages=count,parity_wraps_per_slot=count//slots//2,readiness_checks=checks")
s=s.replace("Path(__file__).with_name('split_ready_cpu_protocol.json')", "Path(__file__).with_name('split_ready_cpu_generations.json')")
s=s.replace("no_gpu_timing_model=True", "explicit_full_and_empty_mbarrier_parity=True,no_gpu_timing_model=True")
(here/'test_split_ready_generations.py').write_text(s,encoding='utf-8')
