"""Load real JIT utilities with package-only adapters for unused server imports."""
import importlib.util,sys,types
from pathlib import Path
import torch
S=Path(__file__).resolve().parents[4]/'sglang/python/sglang'
def prepare():
    for name in ['sglang','sglang.srt','sglang.srt.utils','sglang.kernels','sglang.kernels.jit']:
        path=S.joinpath(*name.split('.')[1:])
        spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    common=types.ModuleType('sglang.srt.utils.common')
    common.is_cuda=lambda:torch.version.cuda is not None
    common.get_cuda_version=lambda:tuple(map(int,torch.version.cuda.split('.')))
    sys.modules[common.__name__]=common
    from sglang.kernels.jit.utils import load_jit
    return load_jit
if __name__=='__main__':
    prepare();print('REAL_JIT_UTILS_IMPORT_OK')
