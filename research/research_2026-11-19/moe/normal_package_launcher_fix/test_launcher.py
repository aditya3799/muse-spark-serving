"""CPU-only tests for fresh-interpreter compiler launch; no CUDA or child execution."""
import ast,builtins,hashlib,json,types
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'muse_spark_grouped_build.py').read_text();tree=ast.parse(source)
assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='CORE_LIMIT_WRAPPER')
wrapper=ast.literal_eval(assignment.value);ast.parse(wrapper)
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compiler_process')
launches=[]
def popen(argv,**kwargs):
    launches.append((argv,kwargs));return types.SimpleNamespace(pid=123)
ns={'sys':types.SimpleNamespace(executable='/python path/python'),
    'subprocess':types.SimpleNamespace(PIPE=-1,STDOUT=-2,Popen=popen),
    'CORE_LIMIT_WRAPPER':wrapper}
exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_compiler_launcher','exec'),ns)
args=['/cuda path/bin/nvcc','-I/include path','-DVALUE=$()','source.cu','-o','kernel.tmp.so']
assert ns['_compiler_process'](args).pid==123
argv,kwargs=launches[0]
assert argv==['/python path/python','-c',wrapper,*args]
assert kwargs==dict(stdout=-1,stderr=-2,text=True,start_new_session=True)
assert 'preexec_fn' not in kwargs and 'shell' not in kwargs
runs=[]
for filter_writable in [True,False]:
    effects=[]
    class Sink:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def write(self,value):effects.append(('filter_write',value))
    def open_filter(path,mode):
        assert path=='/proc/self/coredump_filter' and mode=='w'
        if not filter_writable:raise PermissionError('simulated child-local proc restriction')
        return Sink()
    resource=types.SimpleNamespace(RLIMIT_CORE=4,setrlimit=lambda key,value:effects.append(('limit',key,value)))
    os=types.SimpleNamespace(execv=lambda program,values:effects.append(('execv',program,values)))
    sys=types.SimpleNamespace(argv=['-c',*args])
    modules={'os':os,'resource':resource,'sys':sys}
    env=dict(vars(builtins));env['open']=open_filter
    env['__import__']=lambda name,*a,**kw:modules[name]
    exec(compile(wrapper,'fresh_interpreter_wrapper','exec'),{'__builtins__':env})
    assert effects[0]==('limit',4,(0,0))
    assert effects[-1]==('execv',args[0],args)
    assert any(x[0]=='filter_write' for x in effects)==filter_writable
    runs.append(dict(filter_writable=filter_writable,limit_before_exec=True,argv_exact=True))
# Parent launcher no longer has a Python callback in the post-fork child.
assert not any(isinstance(n,ast.keyword) and n.arg=='preexec_fn' for n in ast.walk(tree))
report=dict(ast=True,command_argument_boundaries_preserved=True,no_shell=True,new_process_session=True,
            no_preexec_callback=True,wrapper_runs=runs,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            scope='Mock Popen and execution of actual wrapper with stub OS/resource; no GPU, compiler, or remote job')
(here/'launcher_tests.json').write_text(json.dumps(report,indent=2));print(report)
