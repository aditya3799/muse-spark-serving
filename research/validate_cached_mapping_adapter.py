from pathlib import Path
import sys
root = Path(r'D:\OpenKernelChallenge\serving')
sys.path.insert(0, str(root))
s = (root/'validate_moe_adapter.py').read_text(encoding='utf-8-sig')
s = s.replace("mods={}\n", '''def fused_align(raw, buckets, block, sorted_ids, experts, count, *, expert_map, local_topk_ids):
 assert buckets == state['local'] + 1
 local_topk_ids.copy_(map_ids(raw, expert_map))
 ordered, expert_order, total = a.align(local_topk_ids, state['local'], block)
 count.copy_(total)
 used = int(total.item())
 sorted_ids[:used].copy_(ordered[:used])
 experts[:used//block].copy_(expert_order[:used//block])
 state['path']='fused'

mods={}
''')
s = s.replace("exports={", "exports={\n 'sglang.kernels.ops.moe.moe_align_small_numel':{'SMALL_NUMEL_LIMIT':64,'moe_align_small_numel':fused_align},")
s = s.replace("('single' if m==1 else 'generic')", "('fused' if m*8<=64 else 'generic')")
s = s.replace("validation/moe_adapter.json", "validation/combined_v4_adapter.json")
s = s.replace("_expert_map=mapping)", "_expert_map=mapping,_expert_lookup=torch.cat((mapping.new_full((1,),-1),mapping)))")
s = s.replace("[1,2,7,16,32,128,513]", "[1,2,7,8,9,16,32,128,513]")
s = s.replace("validation/combined_v4_adapter.json", "validation/v4_cached_mapping_adapter.json")
exec(compile(s,str(root/'validate_moe_adapter.py'),'exec'))
