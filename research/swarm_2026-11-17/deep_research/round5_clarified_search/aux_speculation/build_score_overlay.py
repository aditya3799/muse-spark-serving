"""Generate a review-only default-off sampler export hook, never edit checkout."""
import difflib,hashlib,json
from pathlib import Path
from source_extract import WORKER
P=Path(__file__).parent

def patched_source():
    original=WORKER.read_text()
    start=original.index('class _SelectorDraftSampler:')
    end=original.index('\nclass _DominoDraftSampler:',start)
    old=original[start:end]
    needle='        self.sampling_enabled = sampling_enabled\n'
    assert old.count(needle)==1
    new=old.replace(needle,needle+'        self._research_score_export = None  # Explicit pre-capture attachment only.\n')
    needle='        self.out[: tokens.numel()].copy_(tokens.reshape(-1))\n'
    assert new.count(needle)==1
    new=new.replace(needle,'        if self._research_score_export is not None:\n'
                    '            self._research_score_export.capture_write(candidate_ids, scores, tokens)\n'+needle)
    return original,original[:start]+new+original[end:]

if __name__=='__main__':
    old,new=patched_source()
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
        fromfile='a/python/sglang/srt/speculative/dflash_worker_v2.py',
        tofile='b/python/sglang/srt/speculative/dflash_worker_v2.py'))
    (P/'score_export_overlay.diff').write_text(diff)
    (P/'OVERLAY_SOURCE_MANIFEST.json').write_text(json.dumps({
        'source':str(WORKER),'original_sha256':hashlib.sha256(old.encode()).hexdigest(),
        'patched_sha256':hashlib.sha256(new.encode()).hexdigest(),
        'diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),
        'default_off':True,'worker_controller_hook':'not yet wired; caller must own request/graph lifetime'},indent=2))
