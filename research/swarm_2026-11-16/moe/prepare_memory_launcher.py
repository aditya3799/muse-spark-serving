from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'launch_humming.py').read_text(encoding='utf-8-sig')
s=s.replace("os.environ['MAX_JOBS']='2'", "os.environ['MAX_JOBS']='1'\nos.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'")
s=s.replace("'humming_probe.py'", "'memory_probe.py'")
(p/'launch_memory.py').write_text(s,encoding='utf-8')
import ast
ast.parse((p/'memory_probe.py').read_text(encoding='utf-8-sig'))
print('memory probe prepared; no GPU work executed')
