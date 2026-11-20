from pathlib import Path
p=Path(__file__).resolve().parent
f=p/'grid_experiment.py';s=f.read_text(encoding='utf-8-sig');anchor="        row={'tokens':tokens,'pattern':pattern"
start=s.index(anchor)
insert="""        if not report.get('loaded_binary_provenance'):
            mapped={}
            for entry in Path('/proc/self/maps').read_text().splitlines():
                if 'sgl_kernel_jit_moe_wna16_marlin' in entry and '.so' in entry:
                    binary=Path(entry.split()[-1])
                    if binary.is_file():
                        mapped[str(binary)]=hashlib.sha256(binary.read_bytes()).hexdigest()
            report['loaded_binary_provenance']=mapped
"""
s=s[:start]+insert+s[start:];f.write_bytes(s.replace('\r\n','\n').encode('utf-8'))
f=p/'launch.py';s=f.read_text(encoding='utf-8-sig').replace("os.environ['MAX_JOBS']='2'", "os.environ['MAX_JOBS']='1'");f.write_bytes(s.replace('\r\n','\n').encode('utf-8'))
print('grid loaded-binary provenance enabled; MAX_JOBS1')
