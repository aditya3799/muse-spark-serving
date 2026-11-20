import pathlib,shutil
p=pathlib.Path('serving/research_2026-11-19_round2/moe');s=p/'prove_pr50800_rawscale_layout.py';d=p/'pr50800_rawscale'
shutil.copy2(d/'layout_proof.json',d/'layout_proof_shared_row_helper_initial.json')
shutil.copy2(s,d/'layout_shared_row_helper_initial.py')
t=s.read_text();old='  rows=canonical_row(np.arange(n),up);q=q[:,rows][:,:,kperm(k)];sf=sf[:,rows]'
# Source is one-space indented for this function.
old=old[1:];new=''' if up:
  ki=kperm(4096).reshape(32,2,64)
  rows=np.stack((ki,4096+ki),axis=2).reshape(-1)
 else:rows=np.arange(n)
 q=q[:,rows][:,:,kperm(k)];sf=sf[:,rows]'''
assert old in t;t=t.replace(old,new)
t=t.replace("cpu_only=True,device_decoder_source_present_but_uncompiled=True","cpu_only=True,forward_w13_rows_independent_of_direct_inverse=True,device_decoder_source_present_but_uncompiled=True")
s.write_text(t)
