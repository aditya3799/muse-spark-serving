from pathlib import Path
p=Path(__file__).with_name("gpu_regions.py")
s=p.read_text(encoding="utf8")
s=s.replace('LABEL="early_smoke"','LABEL="qualified_early"')
s=s.replace('"pilot":341593,"holdout":751141','"pilot":915137,"holdout":715631')
s=s.replace('limit=len(r.origin_input_ids) if scene!=SCENES[-1] else p+MIX_N','limit=len(r.origin_input_ids)')
s=s.replace('if scene==SCENES[-1]:\n                reqs+=[short]', 'if scene==SCENES[-1] and p==start:\n                reqs+=[short]')
s=s.replace('FORK+DELTA,MIX_AT+MIX_N','FORK+DELTA,END')
s=s.replace('            if cover:cover.may_publish(r,p+n)','''            if scene==SCENES[-1] and p==start:
                active["short_before"]=workers[arm][1][:,:,perm[-16:]].clone()
            if cover:cover.may_publish(r,p+n)''')
s=s.replace('        return dict(touched=limit-start', '''        if scene==SCENES[-1]:
            assert torch.equal(active["short_before"],workers[arm][1][:,:,perm[-16:]])
        return dict(touched=limit-start''')
s=s.replace('scope+=" Real TpModelWorker','scope+=" Heterogeneous extend runs until both requests can query; checks 16 live short rows and final 2048 live long rows. Real TpModelWorker')
Path(__file__).with_name("gpu_final.py").write_text(s,encoding="utf8")

