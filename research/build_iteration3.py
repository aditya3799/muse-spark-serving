from pathlib import Path
import difflib
root=Path(r'D:\OpenKernelChallenge\serving'); repo=root/'sglang'; changes=[]
rel='python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py'; p=repo/rel
before=p.read_text(encoding='utf-8')
old='''        if num_tokens == 1 and topk <= 32 and expert_map is None:
'''
new='''        # Mapping can merge remote routes into repeated -1 sentinels. The
        # single-token aligner tie-breaks by slot, so these still form a valid
        # inactive prefix while every local route keeps its original slot.
        if num_tokens == 1 and topk <= 32:
'''
assert old in before
s=before.replace(old,new,1); p.write_text(s,encoding='utf-8',newline='\n')
changes.append(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
rel='python/sglang/srt/models/muse_spark.py'; p=repo/rel
before=p.read_text(encoding='utf-8')
old='''        w = self.weight.data
        key = (w.data_ptr(), w.shape, w._version)
'''
new='''        w = self.weight
        # Keep the parameter's version counter: .data hides in-place weight
        # updates and could leave the packed router stale after a reload.
        key = (w.data_ptr(), w.shape, w.dtype, w.device, w._version)
'''
assert old in before
s=before.replace(old,new,1); p.write_text(s,encoding='utf-8',newline='\n')
changes.append(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
(root/'iterations/03_ep_single_token_and_cache.diff').write_text(''.join(changes),encoding='utf-8')
print('Iteration 3: single-token EP alignment and router cache invalidation;',sum(len(s.encode()) for s in changes),'incremental bytes')
