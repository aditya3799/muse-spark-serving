from pathlib import Path
p=Path(r'D:\OpenKernelChallenge\serving\research_2026-11-19_round2\moe\test_facade_memory.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("def inputs(mode):",'''def output_digest(raw,final,ids):
    raw_hash=hashlib.sha256();final_hash=hashlib.sha256();raw_values=0
    for first in range(0,raw.size(0),64):
        live=ids[first:first+64]>=0
        chunk=raw[first:first+64][live]
        raw_values+=chunk.numel()
        raw_hash.update(chunk.view(torch.uint8).cpu().numpy().tobytes())
        final_hash.update(final[first:first+64].view(torch.uint8).cpu().numpy().tobytes())
        del chunk,live
    return dict(raw_sha256=raw_hash.hexdigest(),final_sha256=final_hash.hexdigest(),
                raw_values=raw_values,final_values=final.numel())
def inputs(mode):''')
s=s.replace("# CPU BF16 only: no native output or graph survives into candidate phase.\n        gold[mode]=(y[ids>=0].cpu(),final.cpu())", "# Bounded64-row CPU digest; no complete native output survives this phase.\n        gold[mode]=output_digest(y,final,ids)")
s=s.replace("raw_cpu=y[ids>=0].cpu();final_cpu=final.cpu()\n        assert torch.equal(raw_cpu.view(torch.int16),gold[mode][0].view(torch.int16)),(mode,'raw')\n        assert torch.equal(final_cpu.view(torch.int16),gold[mode][1].view(torch.int16)),(mode,'final')\n        report['correctness'].append(dict(mode=mode,raw_values=raw_cpu.numel(),final_values=final_cpu.numel(),exact=True))\n        del x,ids,scores,y,loc,final,raw_cpu,final_cpu", "observed=output_digest(y,final,ids)\n        assert observed==gold[mode],(mode,observed,gold[mode])\n        report['correctness'].append(dict(mode=mode,**observed,digests_match=True))\n        del x,ids,scores,y,loc,final")
s=s.replace('phase-separated weights and CPU gold; standalone peak counters only','phase-separated weights and bounded64-row CPU SHA256 of live raw/all final BF16 bytes; standalone peak counters only')
p.write_text(s,encoding='utf-8')
