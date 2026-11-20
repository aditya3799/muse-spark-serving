"""Export independent glue-only and persistent-grid candidates from pinned sources."""
import ast, difflib, hashlib, json
from pathlib import Path

P = Path(__file__).resolve().parent
OWN = P.parent
SWARM = OWN.parent
ROOT = SWARM.parent
REL_FA = Path('python/sglang/srt/layers/attention/flashattention_backend.py')
REL_META = Path('python/sglang/kernels/ops/attention/metadata.py')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
scored = SWARM / 'coordination/round5_humming_adaptive_fused/candidate_sources' / REL_FA
optional = OWN / 'combined_attention_optional_v1/candidate' / REL_FA
original = ROOT / 'sglang' / REL_META
assert sha(scored) == '2a720c8f5ed86437e6ec181cace03ae97fba0460672b1d001549f8df4b71e03c'
assert sha(optional) == '0b0d0d4288e7bab6c7c4fa3ec9c1068c75e52ddd019ff9eebf24cb3790b50b9e'
assert sha(original) == '8aa5db57763f5ecfb01a05beba74791d0bf3921952c789d8a8432abf25c231ca'

def export(name, rel, source, text, note):
    out = P / name
    target = out / 'candidate_sources' / rel
    if name == 'glue_only' and target.exists():
        assert target.read_bytes() == text.encode('utf-8')
        print(name, 'frozen export preserved', sha(target))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode('utf-8'))
    old = source.read_text(encoding='utf-8-sig')
    diff = ''.join(difflib.unified_diff(old.splitlines(True), text.splitlines(True), fromfile='a/' + rel.as_posix(), tofile='b/' + rel.as_posix()))
    (out / 'delta_review.diff').write_bytes(diff.encode('utf-8'))
    manifest = dict(status='ISOLATED_SOURCE_CPU_REVIEW_PENDING', note=note, baseline_patch='8b3ed81e053e4ccb2a043afd25cb94bb49017995ece517bb35aeb6e1b79e026b',
                    encoding='UTF-8 LF no BOM', source=str(source), original_sha256=sha(source), candidate_sha256=sha(target),
                    diff_sha256=sha(out / 'delta_review.diff'), scope='No submission; independent candidate, not combined with other new deltas')
    (out / 'manifest.json').write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(name, manifest['candidate_sha256'])

export('glue_only', REL_FA, scored, optional.read_text(encoding='utf-8'),
       'Only narrow glue_draft_metadata eligibility method added to scored backend; all32 existing methods AST-identical; no prefix change')

text = original.read_text(encoding='utf-8')
start = text.index('    col_start = pid_c * BLOCK_COLS\n', text.index('def _fused_decode_metadata_kernel'))
end = text.index('\n\n\n@triton.jit\ndef _draft_extend_metadata_kernel', start)
old_body = text[start:end]
row_line = '    row_idx = tl.load(req_pool_indices + pid_b * req_pool_indices_stride_0)\n'
assert row_line in old_body
body = old_body[old_body.index('    cols = col_start + tl.arange(0, BLOCK_COLS)'):]
loop = (row_line + '    # Preserve every live page (and every tail page when requested), while\n'
        '    # bounding the grid independently of context length. Inputs remain device-fed.\n'
        '    end_col = max_seq_pages if ZERO_TAIL else num_live_pages\n'
        '    # Widen traversal indices so the final stride cannot wrap at int32 bounds.\n'
        '    first_col = pid_c.to(tl.int64) * BLOCK_COLS\n'
        '    column_step = tl.num_programs(1).to(tl.int64) * BLOCK_COLS\n'
        '    for col_start in range(first_col, end_col.to(tl.int64), column_step):\n'
        + ''.join('    ' + line if line.strip() else line for line in body.splitlines(True)))
text = text[:start] + loop + text[end:]
old_grid = '''    grid = (
        (1, 1)
        if skip_page_table or max_seq_pages == 0
        else (batch_size, triton.cdiv(max_seq_pages, block_cols))
    )
'''
new_grid = '''    column_blocks = triton.cdiv(max_seq_pages, block_cols)
    if not skip_page_table and batch_size > 0 and seq_lens.device.type == "cuda":
        # CUDA properties are cached by PyTorch; no tensor values or host lengths
        # are read. Four CTAs per SM bound launch overhead while retaining parallelism.
        sm_count = torch.cuda.get_device_properties(seq_lens.device).multi_processor_count
        column_blocks = min(column_blocks, max(1, triton.cdiv(4 * sm_count, batch_size)))
    grid = (1, 1) if skip_page_table or max_seq_pages == 0 else (batch_size, column_blocks)
'''
assert text.count(old_grid) == 1
text = text.replace(old_grid, new_grid)
ast.parse(text)
export('persistent_metadata', REL_META, original, text,
       'Only bounded per-row CTA page-gather traversal; original scalar prefix loop, translation, masks and ZERO_TAIL preserved; no attention input cropping')
