from pathlib import Path
import difflib,subprocess
root=Path(r'D:\OpenKernelChallenge\serving'); repo=root/'sglang'
rel='python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'; p=repo/rel
before=p.read_text(encoding='utf-8')
old='''            align_num_experts = (
                self.global_num_experts
                if expert_map is not None
                else self.num_local_experts
            )
'''
new='''            # IDs have already been mapped to [-1, num_local_experts).
            # Global-sized histograms waste buckets and can bypass the
            # alignment kernel's small-batch path under expert parallelism.
            align_num_experts = self.num_local_experts
'''
assert old in before
s=before.replace(old,new,1); p.write_text(s,encoding='utf-8',newline='\n')
patch=''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(root/'iterations/02_local_expert_buckets.diff').write_text(patch,encoding='utf-8')
print('Iteration 2:',len(patch.encode()),'incremental patch bytes; EP8 buckets 257 -> 33')
