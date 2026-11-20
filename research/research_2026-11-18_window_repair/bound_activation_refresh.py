from pathlib import Path
p=Path(__file__).with_name("gpu_final.py")
s=p.read_text(encoding="utf8")
s=s.replace('    active={}\n    basis=', '    active={"jitter":0}\n    basis=')
s=s.replace('torch.remainder(active["features"],1024)', 'torch.remainder(active["features"]+active["jitter"],1024)')
s=s.replace('    return reform,run,snapshot,lambda:basis.normal_(generator=gen)','''    def change():
        active["jitter"]=(active["jitter"]+127)%1024
    return reform,run,snapshot,change''')
s=s.replace('faulthandler.dump_traceback_later(45,repeat=True)','faulthandler.dump_traceback_later(90,repeat=True)')
p.write_text(s,encoding="utf8")

