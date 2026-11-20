"""Combined compact transport and metadata-policy Gloo proof; CPU only."""
import os,socket,time
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
import torch
import torch.multiprocessing as mp
import sampler_compact_gloo_probe as transport
import sampler_compact_policy_gloo as policy
torch.set_num_threads(1)
for label,module in [('transport',transport),('startup',policy)]:
 for world in (2,4):
  with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
  start=time.monotonic();mp.start_processes(module.peer,args=(world,port),nprocs=world,join=True,start_method='fork')
  print('PASS compact',label,'TP',world,'seconds',time.monotonic()-start,flush=True)
