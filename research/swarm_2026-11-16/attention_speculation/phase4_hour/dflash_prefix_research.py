"""Research-only target-prefix boundary. Default8 preserves the original path.
Changing this constant is an explicit isolated H200 experiment, not deployment.
All TP ranks must run identical source. This module never changes draft width.
"""
RESEARCH_TARGET_VERIFY_WIDTH = 8

def resolve_verify_prefix(draft_tokens, positions, write_ids, *, eligible, requested_width=None):
    original_width = draft_tokens.shape[1]
    width = RESEARCH_TARGET_VERIFY_WIDTH if requested_width is None else requested_width
    bs = draft_tokens.shape[0]
    if not eligible or original_width != 8 or width not in (1,2,4,6) or bs < 1:
        return original_width, draft_tokens, positions, write_ids, None
    if positions.numel() != bs*8 or write_ids.numel() != bs*8:
        raise ValueError('Full8 draft positions and target write window must agree')
    return (width, draft_tokens[:, :width].contiguous(),
            positions.view(bs,8)[:, :width].contiguous().view(-1),
            write_ids.view(bs,8)[:, :width].contiguous().view(-1), slice(0,width))

def spans_overlap(start_a, size_a, start_b, size_b):
    return size_a>0 and size_b>0 and start_a<start_b+size_b and start_b<start_a+size_a
