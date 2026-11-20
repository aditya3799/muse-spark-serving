"""Check exact original/new C++ control-flow traces without a GPU or a Torch import."""
from pathlib import Path
import ast,hashlib,functools,json,subprocess,types
here=Path(__file__).resolve().parent;root=here.parents[1]
vendor=root/'swarm_2026-11-16/moe/humming_vendor/humming'
node=next(x for x in ast.parse((here/'native_loop.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='source_for')
ns={'Path':Path,'humming':types.SimpleNamespace(__file__=str(vendor/'__init__.py')),'hashlib':hashlib,'lru_cache':functools.lru_cache,'HEADER_SHA':'9a021dffc15f2ba2f3c355b4e508aeb3722b351a8be8130ffd4530743df3edd3'}
exec(compile(ast.Module(body=[node],type_ignores=[]),'native_loop.py','exec'),ns)
original=(vendor/'include/humming/kernel/humming.cuh').read_text()
functions=[]
for v in ('parent','cycles','tail','cycles_tail','countdown'):
 text=original if v=='parent' else ns['source_for'](v)
 (here/(v+'_kernel.cuh')).write_text(text)
 first=text.index('    producer.template load_stage<true, true>(0);');last=text.index('    consumer.wait_channel();',first)
 body=text[first:last]
 functions.append('template<int I,int S,int W> std::vector<Event> '+v+'(){\nusing Ctx=Context<W>; using ProblemShape=Shape<I>;using BlockShape=Shape<1>;constexpr int kNumStages=S;uint32_t slice_iters=I;Recorder rec;Producer producer{rec};Consumer consumer{rec};Pipe s2r_pipe{rec};Math mma{rec};\n'+body+'return rec.events;\n}\n')
source=r'''
#include <vector>
#include <tuple>
#include <cstdint>
#include <cstdio>
#define MAX(a,b) ((a)>(b)?(a):(b))
#define PRAGMA_UNROLL
#define debug_kernel_timeout_check(x)
#define __syncthreads()
using Event=std::tuple<int,int,int>;
struct Recorder{std::vector<Event> events;void add(int f,int a,int b=0){events.emplace_back(f,a,b);}};
struct Producer{Recorder&r;template<bool A=false,bool B=false>void load_stage(int s,bool yes=true){r.add(0,s,yes);}void load_channel(){r.add(1,0);}};
struct Consumer{Recorder&r;template<bool A=false>void wait_stage(int s){r.add(2,s);} };
struct Pipe{Recorder&r;template<bool A=false>void load_stage_iter(int s,int w){r.add(3,s,w);} };
struct Math{Recorder&r;void run(int s,int w){r.add(4,s,w);}void transform_b(int b,int w){r.add(5,b,w);} };
template<int W>struct Context{static constexpr uint32_t kWarpIters=W;static constexpr bool kUseStreamK=false;};
template<int K>struct Shape{static constexpr int value=K;static constexpr int Kvalue=K;};
'''.replace('template<int K>struct Shape{static constexpr int value=K;static constexpr int Kvalue=K;};','template<int V>struct Shape{static constexpr int K=V;};')
source+='\n'.join(functions)
source+=r'''
template<int I,int S,int W> int check(){
 auto p=parent<I,S,W>();auto a=cycles<I,S,W>();auto b=tail<I,S,W>();auto c=cycles_tail<I,S,W>();
 if(p!=a||b!=c||p!=countdown<I,S,W>())return 1;
 int last_mma=-1;for(int i=0;i<(int)p.size();++i)if(std::get<0>(p[i])==4)last_mma=i;
 // One lookahead load immediately precedes the final MMA; one unused transform follows.
 std::vector<Event> expected=p;
 if(last_mma<1||std::get<0>(p[last_mma-1])!=3)return 2;
 int last_transform=-1;for(int i=last_mma+1;i<(int)p.size();++i)if(std::get<0>(p[i])==5){last_transform=i;break;}
 if(last_transform<0)return 3;
 expected.erase(expected.begin()+last_transform);expected.erase(expected.begin()+last_mma-1);
 if(expected!=b)return 4;
 int mm=0;for(auto&e:b)mm+=std::get<0>(e)==4;if(mm!=I*W)return 5;return 0;
}
int main(){int cases=0,errors=0;
'''
for i in (1,2,3,4,5,7,8,15,16,17,31,32,33,63,64,65,96,127,128,129):
 for s in (2,3,4,6):
  for w in (2,4,8):source+='++cases;{int rc=check<%d,%d,%d>();if(rc){++errors;std::printf("FAIL %d %d %d %%d\\n",rc);}}\n'%(i,s,w,i,s,w)
source+='std::printf("CASES %d ERRORS %d\\n",cases,errors);return errors?1:0;}\n'
(here/'trace_control.cpp').write_text(source)
print(json.dumps({'instantiations':240,'source_bytes':len(source),'generated_from_actual_original_and_candidate_cpp':True}))
