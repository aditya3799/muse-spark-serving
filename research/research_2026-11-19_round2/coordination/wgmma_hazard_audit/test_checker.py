import pathlib,ast,importlib.util,json
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("hazard",HERE/"python__sglang__kernels__jit__utils__wgmma_rs_hazard.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
tree=ast.parse((HERE/"test__registered__unit__kernels__test_wgmma_rs_hazard.py").read_text());cases=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CASES' for x in n.targets));result=[]
for name,(expected,body) in cases.items():
 rs,problems=mod.check_sass('Function : '+name+'\n'+body)
 expected_rs=0 if name=='acc_read_in_flight' else body.count('R40, gdesc')+body.count('R44, gdesc')
 passed=len(problems)==expected and rs==expected_rs
 result.append(dict(name=name,expected_problem=bool(expected),rs=rs,problem_count=len(problems),passed=passed))
(HERE/'checker_fixtures.json').write_text(json.dumps(dict(cases=result,all_pass=all(x['passed'] for x in result)),indent=2)+'\n');print(json.dumps(result))
assert all(x['passed'] for x in result)
