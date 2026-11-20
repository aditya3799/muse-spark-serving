import argparse
import ctypes as ct
import importlib
import json
from pathlib import Path
import sys
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    out=p.parse_args().out
    sys.path.insert(0,str(Path(__file__).parent))
    backend=importlib.import_module("native_package_02.hopper_backend")
    lib=backend._library()
    counts=(ct.c_size_t*3)()
    lib.expert_sizes(counts)
    grid=ct.c_int()
    code=lib.expert_config(ct.byref(grid)) if torch.cuda.get_device_capability()==(9,0) else None
    from types import SimpleNamespace
    device=torch.cuda.get_device_capability()
    if device!=(9,0):
        assert backend.HopperExperts.prepare(SimpleNamespace(),SimpleNamespace(w13_weight=torch.empty(1,device="cuda"))) is None
    data=dict(native_extension_path=str(lib._name),capability=device,workspace_elements=list(counts),dynamic_shared_bytes=lib.expert_dynamic_shared_bytes(),
              scratch_bytes=4*(counts[0]+counts[1])+2*counts[2],config_status=code,grid=grid.value)
    (out/"results.json").write_text(json.dumps(data,indent=2))

if __name__=="__main__":
    main()


