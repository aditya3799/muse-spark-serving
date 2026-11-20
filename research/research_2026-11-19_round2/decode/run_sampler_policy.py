"""Reuse the unchanged memory monitor with distinct startup-policy receipts."""
from pathlib import Path
here=Path(__file__).resolve().parent
code=(here/'sampler_gloo_watch.py').read_text()
for before,after in [('sampler_gloo_probe.py','sampler_policy_gloo.py'),('sampler_gloo_stdout.log','sampler_policy_stdout.log'),('sampler_gloo_memory.json','sampler_policy_memory.json')]:code=code.replace(before,after)
exec(compile(code,str(here/'sampler_gloo_watch.py'),'exec'),dict(__file__=__file__,__name__='__main__'))
