from pathlib import Path
import ast
p=Path(__file__).resolve().parent
s=(p/'launch_frozen_probe.py').read_text(encoding='utf-8').replace("'frozen_helper_probe.py'","'offline_sm90_compile.py'")
(p/'launch_offline_sm90.py').write_bytes(s.encode('utf-8'))
f=p/'offline_sm90_compile.py';s=f.read_text(encoding='utf-8-sig');ast.parse(s);f.write_bytes(s.replace('\r\n','\n').encode('utf-8'))
print('offline helper prepared; no rendering/import/compilation executed')
