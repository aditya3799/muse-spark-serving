from pathlib import Path
import ast
here=Path(__file__).resolve().parent
src=(here/'wgs_bridge.py').read_text(encoding='utf-8-sig');tail=src[src.index('def convert('):]
tail=tail.replace("_normal_codes[(n * 2048 // 256, experts)](wgs.view(torch.uint32), codes, n, gate_up, 256)","_normal_tile[(n // 16 * 16, experts)](wgs.view(torch.uint32), codes, n, gate_up, 16, num_warps=4)")
tail=tail.replace('_humming_codes[','_humming_word[')
p=here/'wgs_bridge_tiled.py';text=p.read_text(encoding='utf-8-sig')+'\n\n'+tail;ast.parse(text);p.write_text(text,encoding='utf-8')
original=(here/'test_bridge_gpu.py').read_text(encoding='utf-8-sig');test=original.replace('from wgs_bridge import convert','from wgs_bridge_tiled import convert').replace("'wgs_bridge.py'","'wgs_bridge_tiled.py'").replace('bridge_gpu.json','bridge_tiled_gpu.json')
(here/'test_bridge_tiled_gpu.py').write_text(test,encoding='utf-8')
print('AST/isolated test generation passed')
