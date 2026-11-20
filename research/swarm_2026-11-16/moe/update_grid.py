from pathlib import Path
p=Path('/mnt/d/OpenKernelChallenge/serving/swarm_2026-11-16/moe/grid_experiment.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("p.add_argument('--samples',type=int,default=5)", "p.add_argument('--samples',type=int,default=5);p.add_argument('--caps',type=int,nargs='+',default=[1,2])")
s=s.replace("for cap in (1,2)}", "for cap in args.caps}")
anchor="    torch.manual_seed(101)"
assert anchor in s
s=s.replace(anchor,"    print('Preparing grid experiment weights',flush=True)\n"+anchor)
anchor="    report_path=args.out/'results.json'"
s=s.replace(anchor,anchor+"\n    mapped=[]\n    for line in Path('/proc/self/maps').read_text().splitlines():\n        if 'sgl_kernel_jit_moe_wna16_marlin' in line and '.so' in line:\n            path=line.split()[-1]\n            if path not in [r['path'] for r in mapped]:\n                mapped.append({'path':path,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()})\n    report['loaded_binary_provenance']=mapped")
p.write_text(s)
print('grid caps selector and binary provenance added')
