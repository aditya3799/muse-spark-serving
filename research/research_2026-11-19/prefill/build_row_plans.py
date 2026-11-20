from pathlib import Path
import subprocess,sysconfig
here=Path(__file__).resolve().parent
nvidia=Path(sysconfig.get_paths()['purelib'])/'nvidia'
location=next(p for p in (nvidia/'cu13',nvidia/'cublas') if (p/'include/cublasLt.h').is_file())
library=next((location/'lib').glob('libcublasLt.so.*'))
command=['g++','-O2','-std=c++17','-shared','-fPIC']
for p in (location,nvidia/'cuda_runtime'):
    if (p/'include').is_dir():command+=['-I',str(p/'include')]
command += [str(here/'row_plans.cc'),str(library),'-Wl,-rpath,'+str(location/'lib'),'-o',str(here/'row_plans.so')]
subprocess.run(command,check=True)
print(here/'row_plans.so')
