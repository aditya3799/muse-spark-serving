from pathlib import Path
p=Path(__file__).with_name("gpu_regions.py");s=p.read_text(encoding="utf8")
s=s.replace('        for scene in SCENES:\n            d=detail["cases"][scene]', '        for scene in SCENES:\n            cold_times={}\n            d=detail["cases"][scene]')
s=s.replace('                if step in (-1,0):print(stage,step,scene,arm,round(us/1e6,3),flush=True)','                cold_times[arm]=us\n                if step in (-1,0):print(stage,step,scene,arm,round(us/1e6,3),flush=True)')
s=s.replace('            if step>=0:\n                base=snapshot', '''            if step==-1:
                ref=snapshot("baseline",scene)
                warm=detail.setdefault("unscored_warmup",{})
                warm[scene]=dict(synchronized_wall_us=cold_times,compared=ref.numel(),
                    wrong={arm:torch.count_nonzero(snapshot(arm,scene)!=ref).item() for arm in ARMS[1:]})
                (HERE/f"{LABEL}_{stage}_warmup.json").write_text(json.dumps(warm,indent=2))
            if step>=0:
                base=snapshot''')
p.write_text(s,encoding="utf8")

