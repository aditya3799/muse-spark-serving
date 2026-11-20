from pathlib import Path
import ast,copy,json
P=Path(__file__).resolve().parent;OLD=P.parents[1]/'hopper_moe';owned=json.loads((OLD/'grid_body1_retry/owned_configs.json').read_text());oldjobs=json.loads((P/'SM86_REUSE_JOBS.json').read_text())['jobs'];heavy=copy.deepcopy(oldjobs[0]);heavy['label']='heavy';light=copy.deepcopy(heavy);light['label']='light';light['configuration'].update(next(c for lo,hi,c in owned['indexed']['w13'] if lo<32<=hi));light['configuration'].pop('num_sms',None);m2=copy.deepcopy(oldjobs[2]);m2['label']='m2';jobs=[heavy,light,m2]
(P/'LEGAL_M32_JOBS.json').write_text(json.dumps({'scope':'Actual scored native SM86 M32 W13 light/heavy; only heavy eligible routes may use exact M2.','jobs':jobs},indent=2)+'\n',encoding='utf-8')
s=(P/'probe_sm86_reuse.py').read_text(encoding='utf-8-sig');s=s[:s.index(' for m in args.tokens:')]
s=s.replace("default=[8,128]","default=[32]").replace("default=['uniform','hot_expert']","default=['uniform','hot_one_local','hot_eight_local','mixed','all_remote']")
s=s.replace("any(m<=0 or m>256 for m in args.tokens)","args.tokens!=[32]")
s=s.replace("P/'SM86_REUSE_JOBS.json'","P/'LEGAL_M32_JOBS.json'")
s=s.replace("'source_hashes':{str(f):sha(f) for f in files}","'source_hashes':{str(f):sha(f) for f in files}")
s=s.replace("files=[Path(__file__),","files=[P/'configuration_contract.py',P/'selective_routes.py',Path(__file__),")
s=s.replace("Fixed W13 configuration; compare kernel-only and GPU alignment+GEMM+BF16 scale+SiLU.","Actual scored native M32 W13 light/heavy policy versus three partitions. Includes GPU histogram, all alignment and GEMM launches, BF16 scale and SiLU.")
(P/'probe_legal_selective_m2.py').write_text(s+(P/'legal_selective_tail.py.source').read_text(encoding='utf-8-sig'),encoding='utf-8');ast.parse((P/'probe_legal_selective_m2.py').read_text(encoding='utf-8'));print('Prepared actual-policy M32 legal gate, no runtime')
