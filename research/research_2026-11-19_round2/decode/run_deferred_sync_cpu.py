"""Unchanged guarded Gloo monitor with a separate artifact namespace."""
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'sampler_gloo_watch.py').read_text().replace('sampler_gloo_probe.py','deferred_sync_gloo.py').replace('sampler_gloo_stdout.log','deferred_sync_gloo_stdout.log').replace('sampler_gloo_memory.json','deferred_sync_gloo_memory.json')
exec(compile(source,str(here/'sampler_gloo_watch.py'),'exec'))
