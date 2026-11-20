"""Select bounded context-free compile through the durable phase2 supervisor."""
from pathlib import Path
import os,sys
P=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MOE_OFFLINE_TIMEOUT_SECONDS']='240'
os.execv(sys.executable,[sys.executable,str(P/'launch_probe.py'),'--script','offline_grouped_sm90.py','--seconds','240',*sys.argv[1:]])
