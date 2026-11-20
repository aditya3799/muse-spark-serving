from pathlib import Path
h=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/prefill')
s=(h/'prefill_prologue_bytes.py').read_text(encoding='utf-8-sig')
anchor="    assert all(checks.values()),records[-1]"
replacement="""    if not all(checks.values()):
     live=loc!=0
     unique=loc[live].unique().numel()==int(live.sum())
     # Independent direct scatter oracle: same unique, nonzero destinations.
     direct=initial[0].clone()
     expected_k=k[:n].to(torch.float8_e4m3fn).view(torch.uint8)
     direct[loc[live]]=expected_k[live]
     q_real,k_real=native_norm(norm,x[:n],pos[:n] if sliding else None,rope if sliding else None,kv_store=None)
     real_k8=k_real.to(torch.float8_e4m3fn).view(torch.uint8)
     mism=(actual[0].view(torch.uint8)!=gold[0]).nonzero()
     samples=[]
     for cache_row,col in mism[:32].cpu().tolist():
      input_rows=(loc==cache_row).nonzero().flatten()
      token=int(input_rows[0]) if input_rows.numel() else None
      sample=dict(cache_row=cache_row,column=col,token=token,
        candidate_byte=int(actual[0].view(torch.uint8)[cache_row,col]),
        native_store_byte=int(gold[0][cache_row,col]),direct_store_byte=int(direct[cache_row,col]))
      if token is not None:
       sample.update(native_padded_bf16=float(k[token,col]),native_real_bf16=float(k_real[token,col]),
         native_real_fp8_byte=int(real_k8[token,col]),
         input_head=x[token,1024+(col//64)*64:1024+(col//64+1)*64].float().cpu().tolist())
      samples.append(sample)
     diagnostics=dict(mismatch_count=int(mism.shape[0]),destinations_unique=unique,
       location_min=int(loc.min()),location_max=int(loc.max()),capacity=cap,
       native_store_equals_direct=torch.equal(gold[0],direct),
       candidate_equals_direct=torch.equal(actual[0].view(torch.uint8),direct),
       native_real_equals_native_padded_bf16=torch.equal(k_real.view(torch.int16),k[:n].view(torch.int16)),
       native_real_equals_native_padded_fp8=torch.equal(real_k8,expected_k),samples=samples)
     failure=dict(scope=__doc__,passed=False,records=records,first_failure=records[-1],diagnostics=diagnostics,
       peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
       elapsed_s=time.perf_counter()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       kernel_sha256=hashlib.sha256((DECODE/'prologue_package/kernel.py').read_bytes()).hexdigest())
     (HERE/'prefill_prologue_diagnostic.json').write_text(json.dumps(failure,indent=2))
     print(json.dumps(dict(failure=records[-1],diagnostics=diagnostics),indent=2),flush=True)
     return
"""
assert s.count(anchor)==1
s=s.replace(anchor,replacement).replace("(HERE/'prefill_prologue_bytes.json')","(HERE/'prefill_prologue_diagnostic.json')")
(h/'prefill_prologue_diagnostic.py').write_text(s)
