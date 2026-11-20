from pathlib import Path
import ast
p=Path(__file__).resolve().parent
f=p/'launch_offline_sm90.py';s=f.read_text(encoding='utf-8')
s=s.replace("os.execv(sys.executable,[sys.executable,str(script)]+sys.argv[1:])", "os.environ['MOE_OFFLINE_TIMEOUT_SECONDS']='240'\nos.execv('/usr/bin/timeout',['/usr/bin/timeout','--signal=TERM','--kill-after=15s','240s',sys.executable,str(script)]+sys.argv[1:])")
ast.parse(s);f.write_bytes(s.encode('utf-8'))
f=p/'offline_sm90_compile.py';s=f.read_text(encoding='utf-8')
s=s.replace("if a.tokens<=0:raise ValueError('Positive token count required')", "if a.tokens<=0:raise ValueError('Positive token count required')\nif os.environ.get('MOE_OFFLINE_TIMEOUT_SECONDS')!='240':\n    raise RuntimeError('Use launch_offline_sm90.py for the bounded TERM/KILL process group')")
s=s.replace("'architecture':'sm_90a',", "'architecture':'sm_90a','process_timeout':{'wall_seconds':240,'kill_after_seconds':15,'executable':'/usr/bin/timeout'},")
ast.parse(s);f.write_bytes(s.encode('utf-8'));print('offline process group bounded by TERM240s/KILL15s; no launch')
