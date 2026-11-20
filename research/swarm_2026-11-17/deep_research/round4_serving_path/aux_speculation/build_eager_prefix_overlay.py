"""Generate a reviewable, inert source overlay; never mutate the checkout."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'eager_prefix_overlay'


def once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


def replace_function(source, cls, name, transform):
    tree = ast.parse(source)
    owner = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    fn = next(x for x in owner.body if isinstance(x, ast.FunctionDef) and x.name == name)
    lines = source.splitlines(keepends=True)
    old = ''.join(lines[fn.lineno-1:fn.end_lineno])
    new = transform(old)
    return ''.join(lines[:fn.lineno-1]) + new + ''.join(lines[fn.end_lineno:])


def worker_change(source):
    source = once(source, 'import torch\n', 'import torch\n\nfrom sglang.srt.speculative.dflash_prefix_verification import (\n    next_prefix_accept_buffers,\n    resolve_prefix_verification_width,\n)\n')
    def buffers(text):
        text = once(text, 'self, bs: int\n', 'self, bs: int, block_size: Optional[int] = None\n')
        return once(text, '        self._ensure_accept_bonus_buffers(bs)\n', '        if block_size is not None and block_size != int(self.block_size):\n            return next_prefix_accept_buffers(self, bs, block_size)\n        self._ensure_accept_bonus_buffers(bs)\n')
    source = replace_function(source, 'DFlashWorkerV2', '_next_accept_bonus_buffers', buffers)
    def accept(text):
        text = once(text, '    ):\n        new_seq_lens = None\n', '''    ):
        if candidates.shape[1] != int(self.block_size):
            expected_rows = bs * candidates.shape[1]
            if target_predict is not None:
                if target_predict.numel() != expected_rows:
                    raise ValueError("DFLASH prefix predictions retain padding")
            elif next_token_logits.shape[0] != expected_rows:
                raise ValueError("DFLASH prefix logits retain padding")
        new_seq_lens = None
''')
        text = once(text, 'target_predict.view(bs, int(self.block_size))', 'target_predict.view(bs, candidates.shape[1])')
        return once(text, 'self._next_accept_bonus_buffers(bs)', 'self._next_accept_bonus_buffers(bs, candidates.shape[1])')
    source = replace_function(source, 'DFlashWorkerV2', '_accept_block', accept)
    def forward(text):
        marker = '        # --- 2) Target verify.\n'
        before, after = text.split(marker)
        prefix = '''        # Draft diffusion and proposal generation retain their original width.
        verify_block_size = resolve_prefix_verification_width(
            self, batch, get_parallel()
        )
        verify_cols = None
        if verify_block_size != block_size:
            draft_tokens = draft_tokens[:, :verify_block_size].contiguous()
            positions = positions_2d[:, :verify_block_size].contiguous().view(-1)
            verify_out_cache_loc_2d = (
                verify_out_cache_loc_2d[:, :verify_block_size].contiguous()
            )
            verify_out_cache_loc = verify_out_cache_loc_2d.view(-1)
            verify_cols = slice(0, verify_block_size)

'''
        after = once(after, '            kv_loc_plan=kv_loc_plan,\n', '            kv_loc_plan=kv_loc_plan,\n            kv_loc_cols=verify_cols,\n')
        # All occurrences after this boundary are target-derived widths, except
        # result.speculative_num_draft_tokens, which remains proposal accounting.
        after = after.replace('draft_token_num=int(self.block_size)', 'draft_token_num=verify_block_size')
        after = once(after, 'verify_host_seq_lens = seq_lens_cpu_backup + block_size', 'verify_host_seq_lens = seq_lens_cpu_backup + verify_block_size')
        after = once(after, 'chain_stride=block_size', 'chain_stride=verify_block_size')
        after = once(after, '        hidden = hidden.view(bs, int(self.block_size), -1)\n', '''        if verify_block_size != block_size and (
            hidden.ndim != 2 or hidden.shape[0] != bs * verify_block_size
        ):
            raise ValueError("DFLASH prefix aux rows retain padding")
        hidden = hidden.view(bs, verify_block_size, -1)
''')
        after = once(after, '        cache_loc = self._draft_write_ids(kv_loc_plan)\n', '        cache_loc = kv_loc_plan.write_ids(\n            self.draft_model_runner.kv_index_translator, cols=verify_cols\n        )\n')
        after = once(after, '            speculative_num_draft_tokens=int(self.block_size),\n', '            speculative_num_draft_tokens=int(self.block_size),\n            speculative_output_stride=(\n                verify_block_size if verify_block_size != block_size else None\n            ),\n')
        return before + prefix + marker + after
    return replace_function(source, 'DFlashWorkerV2', 'forward_batch_generation', forward)


def fa3_change(source):
    source = once(source, 'from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm', 'from sglang.srt.speculative.spec_info import (\n    SpecInput,\n    SpecInputType,\n    SpeculativeAlgorithm,\n)')
    def eager(text):
        start = '        elif forward_batch.forward_mode.is_target_verify():\n'
        begin, tail = text.split(start, 1)
        # Only the uniform topk1 eager branch. Ragged/tree paths retain stock.
        a = tail.index('                else:\n                    metadata.cache_seqlens_int32 = (')
        b = tail.index('                metadata.page_table = ', a)
        branch = tail[a:b]
        assert branch.count('self.speculative_num_draft_tokens') == 5
        branch = branch.replace('self.speculative_num_draft_tokens', 'verify_width')
        tail = tail[:a] + branch + tail[b:]
        addition = '''            verify_width = self.speculative_num_draft_tokens
            if (
                forward_batch.spec_info is not None
                and forward_batch.spec_info.spec_input_type
                == SpecInputType.DFLASH_VERIFY
            ):
                verify_width = int(forward_batch.spec_info.draft_token_num)
                if verify_width <= 0:
                    raise ValueError("DFLASH verification width must be positive")
'''
        return begin + start + addition + tail
    return replace_function(source, 'FlashAttentionBackend', 'init_forward_metadata', eager)


def main():
    transforms = {
        'python/sglang/srt/speculative/dflash_worker_v2.py': worker_change,
        'python/sglang/srt/layers/attention/flashattention_backend.py': fa3_change,
    }
    diffs = []
    report = {'status': 'INERT_EAGER_ONLY_OVERLAY', 'active_checkout_changed': False,
              'activation': 'Explicit fixture call configure_prefix_verification(worker,W); no server flag or automatic policy',
              'graph_policy': 'Existing width8 mismatch guard unchanged; shorter target width must use eager runner',
              'sources': {}, 'output': {}, 'holds': ['Full model arithmetic/greedy agreement', 'Actual FA3 GPU execution', 'Target exact-width graph runners and pool accounting', 'TP/EP collective agreement and rank-consistent configuration']}
    for rel, transform in transforms.items():
        source = (ROOT/'sglang'/rel).read_text()
        result = transform(source)
        compile(result, rel, 'exec')
        target = OUT/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result)
        report['sources'][rel] = hashlib.sha256(source.encode()).hexdigest()
        report['output'][rel] = hashlib.sha256(result.encode()).hexdigest()
        diffs.extend(difflib.unified_diff(source.splitlines(True), result.splitlines(True), fromfile='a/'+rel, tofile='b/'+rel))
    helper_rel = 'python/sglang/srt/speculative/dflash_prefix_verification.py'
    helper = (HERE/'dflash_prefix_verification.py').read_text(encoding='utf-8-sig')
    compile(helper, helper_rel, 'exec')
    (OUT/helper_rel).write_text(helper)
    report['output'][helper_rel] = hashlib.sha256(helper.encode()).hexdigest()
    diffs.extend(difflib.unified_diff([], helper.splitlines(True), fromfile='/dev/null', tofile='b/'+helper_rel))
    patch = ''.join(diffs)
    (HERE/'eager_prefix_overlay.diff').write_text(patch)
    report['diff_bytes'] = len(patch.encode())
    report['max_added_line_length'] = max(len(line[1:]) for line in patch.splitlines() if line.startswith('+'))
    report['added_lines'] = sum(line.startswith('+') and not line.startswith('+++') for line in patch.splitlines())
    report['removed_lines'] = sum(line.startswith('-') and not line.startswith('---') for line in patch.splitlines())
    (HERE/'EAGER_PREFIX_OVERLAY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()


