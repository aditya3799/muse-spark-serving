"""Launch the locally compiled SM86 verification-tree kernel from the Windows PyTorch runtime."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent

class CompiledPlanner:
    def __init__(self, kind="planner"):
        if torch.cuda.get_device_capability() != (8, 6):
            raise ValueError("this benchmark module was compiled for SM86")
        torch.cuda.current_stream()
        self.driver = ctypes.WinDLL("nvcuda.dll")
        self.abi = json.loads((HERE/(kind+"_abi.json")).read_text())
        self.width = self.abi.get("nodes",8)
        binary = (HERE/(kind+"_sm86.cubin")).read_bytes()
        assert hashlib.sha256(binary).hexdigest() == self.abi["sha256"]
        self.binary = ctypes.create_string_buffer(binary)
        self.module = ctypes.c_void_p()
        self.func = ctypes.c_void_p()
        self.check(self.driver.cuModuleLoadData(ctypes.byref(self.module), self.binary))
        self.check(self.driver.cuModuleGetFunction(ctypes.byref(self.func), self.module,
                                                  self.abi["symbol"].encode()))
        self.storage = {}
        self.zero_mass = {}

    def check(self, result):
        if result:
            message = ctypes.c_char_p()
            self.driver.cuGetErrorString(result, ctypes.byref(message))
            raise RuntimeError(f"CUDA driver status {result} {message.value!r}")

    def build(self, candidates, scores, anchors, temperature, mass, fork):
        b = candidates.shape[0]
        assert candidates.dtype == anchors.dtype == torch.int64
        assert candidates.shape == (b, 7, 16) and scores.shape == (b, 7, 16, 16)
        scores = scores.float().contiguous()
        if mass is None:
            if b not in self.zero_mass:
                self.zero_mass[b] = torch.zeros((b,7),device=candidates.device)
            mass = self.zero_mass[b]
        inputs = (candidates,scores,mass,anchors)
        assert all(t.is_cuda and t.is_contiguous() and t.data_ptr()%16==0 for t in inputs)
        assert mass.dtype == torch.float32
        if b not in self.storage:
            self.storage[b] = tuple(torch.empty((b,self.width),dtype=tp,device=candidates.device)
                                    for tp in (torch.int64,torch.int32,torch.int32,torch.uint32))
        outputs = self.storage[b]
        args = [ctypes.c_void_p(t.data_ptr()) for t in (*inputs,*outputs)]
        args += [ctypes.c_float(1/temperature),ctypes.c_float(math.log(fork))]
        args.extend(ctypes.c_void_p() for _ in self.abi["params"][len(args):])
        self.launch(args,b)
        return outputs

    def launch(self,args,b):
        argptr = (ctypes.c_void_p * len(args))(*(ctypes.addressof(v) for v in args))
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(self.driver.cuLaunchKernel(self.func, b,1,1, self.abi["threads"],1,1,
                                              self.abi["shared"],stream,argptr,None))

    def single(self,candidates,scores,anchor,policy,mass):
        _,temp,covered,fork = policy.split("_")
        ids,parents,depth,bits = self.build(candidates,scores,anchor,float(temp[1:]),
                                            mass if int(covered[1:]) else None,float(fork[1:]))
        offset = torch.arange(self.width,device=bits.device)
        mask = ((bits[0].long()[:,None] >> offset[None]) & 1).bool()
        return ids[0],parents[0].tolist(),depth[0],mask

    def chain(self,candidates,scores,anchor):
        b = candidates.shape[0]
        assert b == 1 and scores.dtype == torch.bfloat16
        if b not in self.storage:
            self.storage[b] = (torch.zeros((b,7),device=scores.device),torch.ones(b,device=scores.device),
                torch.ones(b,device=scores.device,dtype=torch.int32),
                torch.empty((b,7),device=scores.device,dtype=torch.int64),
                torch.empty((b,7,16),device=scores.device,dtype=torch.float32),
                torch.arange(8,device=scores.device))
        uniforms,temperatures,greedy,tokens,q,pos = self.storage[b]
        args = [ctypes.c_void_p(t.data_ptr()) for t in (scores,candidates,uniforms,temperatures,greedy,tokens,q)]
        args.extend(ctypes.c_void_p() for _ in self.abi["params"][len(args):])
        self.launch(args,b)
        return torch.cat((anchor,tokens[0])),list(range(-1,7)),pos,None