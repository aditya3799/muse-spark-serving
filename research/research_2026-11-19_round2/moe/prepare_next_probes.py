from pathlib import Path
import ast
here=Path(__file__).resolve().parent
p=here/'compile_wgs_register.py';s=p.read_text(encoding='utf-8-sig').replace('3.3*1024**3','2.3*1024**3').replace('3*1024**3','2*1024**3')
s=s.replace("os.execv(sys.argv[1],sys.argv[1:])", "os.execv(sys.argv[1],sys.argv[1:])")
s=s.replace("[sys.executable,'-c',wrapper,*cmd]", "[sys.executable,'-c',wrapper,'/usr/bin/time','-v',*cmd]")
p.write_text(s,encoding='utf-8');ast.parse(s)
s=(here/'bench_hybrid_fallback.py').read_text(encoding='utf-8-sig')
s=s.replace('from hybrid_facade import SharedWeightWorkspace','from hybrid_facade import SharedWeightWorkspace\nimport wgs_bridge\nfrom wgs_bridge_tiled import convert as tiled_convert\noriginal_convert=wgs_bridge.convert')
s=s.replace("'wgs_bridge.py'","'wgs_bridge_tiled.py'")
s=s.replace("f'fallback_{args.seed}.json'","f'tiled_full_ffn_{args.seed}.json'")
s=s.replace("WGS->native Humming conversion + original W13/scale/SiLU/W2/finalizer versus original canonical Humming; SM86 fallback cost, not native H200 WGS performance","Tiled versus generic WGS restoration, both including identical complete W13/scale/SiLU/W2/finalizer. Parent correctness reference uses canonical weights. LocalSM86 only.")
old="if which=='candidate':y,loc=workspace.project(owner,helper,x,ids,defer_downscale=True,global_experts=256)\n            else:y,loc=helper(parent,x,ids,torch,defer_downscale=True,global_experts=256)"
new="if which=='parent':y,loc=helper(parent,x,ids,torch,defer_downscale=True,global_experts=256)\n            else:\n                wgs_bridge.convert=tiled_convert if which=='candidate' else original_convert\n                y,loc=workspace.project(owner,helper,x,ids,defer_downscale=True,global_experts=256)"
assert old in s;s=s.replace(old,new);ast.parse(s)
(here/'bench_hybrid_tiled.py').write_text(s,encoding='utf-8')
p=here/'launch.py';s=p.read_text(encoding='utf-8-sig');s=s.replace("'bench_hybrid_fallback'", "'bench_hybrid_fallback','test_bridge_tiled_gpu','bench_hybrid_tiled'");ast.parse(s);p.write_text(s,encoding='utf-8')
print('prepared bounded compiler and meaningful complete-FFN tiled comparison')
