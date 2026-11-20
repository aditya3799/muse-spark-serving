from pathlib import Path
P=Path(__file__).resolve().parent;p=P/'legal_selective_tail.py.source';s=p.read_text(encoding='utf-8-sig')
s=s.replace(" from selective_routes import partition3"," from selective_routes import partition3\n from configuration_contract import m2_compatible")
s=s.replace("for which in ('A','B')","for which in ('A','B','C')")
s=s.replace(" configs={j['label']:j['configuration'] for j in jobs}"," configs={j['label']:j['configuration'] for j in jobs}\n compatible=m2_compatible(configs['heavy'],configs['m2'],cutoff)\n report['metadata_M2_compatible']=compatible")
s=s.replace("  else:\n   light,heavy,m2,_=partition3", "  elif which=='B':\n   light,heavy,_=split_module.partition_routes(ids,e,cutoff);launch('light',light,which);launch('m2' if compatible else 'heavy',heavy,which)\n  else:\n   light,heavy,m2,_=partition3")
s=s.replace("    assert exact(buffers['A'][active],buffers['B'][active]),(mode,'bitwise')", "    assert exact(buffers['A'][active],buffers['B'][active]) and exact(buffers['A'][active],buffers['C'][active]),(mode,'bitwise')")
s=s.replace("torch.isfinite(buffers['B'][active]).all())","torch.isfinite(buffers['B'][active]).all() and torch.isfinite(buffers['C'][active]).all())")
s=s.replace("candidate_GEMM_launches=3,baseline_GEMM_launches=2", "candidate_GEMM_launches=2,conservative_ablation_GEMM_launches=3,baseline_GEMM_launches=2")
a=s.index('  for cold in (False,True):');b=s.index("  print(json.dumps",a)
old=s[a:b]
old=old.replace("for which in ('A','B','C')", "for which in ('A',variant)")
old=old.replace("graphs[arm[0]]", "graphs[variant if arm.startswith('B') else 'A']")
old=old.replace("row['timing']['cold' if cold else 'warm']", "row['timing'][variant+('_cold' if cold else '_warm')]")
new="  for variant in ('B','C'):\n"+'\n'.join(' '+line for line in old.rstrip().splitlines())+'\n'
s=s[:a]+new+s[b:]
p.write_text(s,encoding='utf-8')
p=P/'build_legal_probe.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("files=[P/'selective_routes.py',Path(__file__),", "files=[P/'configuration_contract.py',P/'selective_routes.py',Path(__file__),");p.write_text(s,encoding='utf-8')
