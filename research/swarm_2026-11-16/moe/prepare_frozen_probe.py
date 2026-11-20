from pathlib import Path
import ast
p=Path(__file__).resolve().parent
s=(p/'standalone_probe.py').read_text(encoding='utf-8-sig')
s=s[:s.index('def prepare_humming')]+s[s.index('def main():'):]
s=s.replace("p.add_argument('--import-only',action='store_true')", "p.add_argument('--different-blocks',action='store_true');p.add_argument('--import-only',action='store_true')")
old="    spec=importlib.util.spec_from_file_location('standalone',HERE/'submission_humming_fp32_v1.py')\n    standalone=importlib.util.module_from_spec(spec);spec.loader.exec_module(standalone)"
new="""    export=HERE/'serving_humming_v1/python/sglang/srt/layers/muse_spark/muse_spark_nvfp4_moe.py'
    frozen_tree=ast.parse(export.read_text(encoding='utf-8'))
    helper_nodes=[n for n in frozen_tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_prepare_humming_experts','_humming_expert_outputs')]
    helper_ns={'torch':torch}
    exec(compile(ast.Module(body=helper_nodes,type_ignores=[]),str(export),'exec'),helper_ns)
"""
assert old in s;s=s.replace(old,new)
s=s.replace("    state=standalone.prepare(weights)", "    state={'layer':helper_ns['_prepare_humming_experts'](raw,torch),'norm_w':(norm.float()+1).to(torch.bfloat16)}")
s=s.replace("'scope':'standalone exact file; K4096 scale stress, changed graph inputs, FP32 accumulation and no Stream-K', 'artifact_sha256':hashlib.sha256((HERE/'submission_humming_fp32_v1.py').read_bytes()).hexdigest()", "'scope':'exact frozen serving helpers; scaled globals, changed graph inputs, FP32 accumulation and no Stream-K', 'artifact_sha256':hashlib.sha256(export.read_bytes()).hexdigest(), 'different_blocks':args.different_blocks")
old="        def candidate():return standalone.kernel_fn({'hidden_states':x,'topk_ids':ids,'topk_weights':scores,'config':{'local_start':0,'local_count':args.experts,'top_k':8}},state)"
new="""        if args.different_blocks:
            first=next(c for lo,hi,c in humm.fp32_configs['w13'] if lo<ids.numel()<=hi)
            different=next(c for lo,hi,c in humm.fp32_configs['w2'] if c['block_shape'][0]!=first['block_shape'][0])
            humm.fp32_configs['w2']=[(0,1<<30,different)]
        def candidate():return finish(*helper_ns['_humming_expert_outputs'](humm,x,raw._expert_lookup[ids+1],torch))"""
assert old in s;s=s.replace(old,new)
s=s.replace("    report['passed']=all", "        if (args.out/'STOP_AFTER_CASE').exists():\n            report['stopped_cooperatively']=True\n            (args.out/'results.json').write_text(json.dumps(report,indent=2))\n            return\n    report['passed']=all")
s=s.replace("    for m in args.tokens:", "    original_w2_configs=humm.fp32_configs['w2']\n    for m in args.tokens:\n      humm.fp32_configs['w2']=original_w2_configs")
# A single token group is recommended; preserve original stage2 candidates across patterns.
s=s.replace("different=next(c for lo,hi,c in humm.fp32_configs['w2']", "different=next(c for lo,hi,c in original_w2_configs")
ast.parse(s);(p/'frozen_helper_probe.py').write_bytes(s.replace('\r\n','\n').encode('utf-8'))
l=(p/'launch_humming.py').read_text(encoding='utf-8-sig').replace("os.environ['MAX_JOBS']='2'", "os.environ['MAX_JOBS']='1'\nos.environ['HUMMING_DISABLE_PARALLEL_BUILD']='1'").replace("'humming_probe.py'", "'frozen_helper_probe.py'")
(p/'launch_frozen_probe.py').write_bytes(l.replace('\r\n','\n').encode('utf-8'))
print('exact frozen-helper probe prepared; no execution')
