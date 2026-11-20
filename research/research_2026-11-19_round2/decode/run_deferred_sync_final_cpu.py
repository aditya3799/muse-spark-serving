"""Final native-sync guard validation, preserving first-pass receipts."""
import os
from pathlib import Path
here=Path(__file__).resolve().parent
os.environ['DEFERRED_SYNC_TAG']='_final'
source=(here/'sampler_gloo_watch.py').read_text().replace('sampler_gloo_probe.py','deferred_sync_gloo.py').replace('sampler_gloo_stdout.log','deferred_sync_final_gloo_stdout.log').replace('sampler_gloo_memory.json','deferred_sync_final_gloo_memory.json')
exec(compile(source,str(here/'sampler_gloo_watch.py'),'exec'))
