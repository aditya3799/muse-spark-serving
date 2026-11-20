"""Source-derived four-block fixture, exact SM86 SWA origin on all arms."""
import ast,hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
old=root/'research_2026-11-18_rl_prefill/decoder_chain/bench.py'
s=old.read_text()
def once(before,after):
 global s
 assert s.count(before)==1,(before[:100],s.count(before))
 s=s.replace(before,after)
once('import operators as op',f"sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'research_2026-11-18_rl_prefill/decoder_chain'))\nimport operators as op\nfrom chain_entry import install\ninstall(op)")
once('candidate="packet+swa_schedule+grouped_at_supported_M128",','candidate="guarded_native18_prologue_only",')
once('report=dict(agent="prefill",','report=dict(agent="decode",')
once('source=dict(fa=sha(inspect.getfile(fa)),', '''source=dict(kernel=sha(Path(__file__).parent/'prologue_package/kernel.py'),
                    adapter=sha(Path(__file__).parent/'prologue_package/runtime.py'),
                    entry=sha(Path(__file__).parent/'chain_entry.py'),fixture=sha(__file__),
                    fa=sha(inspect.getfile(fa)),''')
once('candidate additionally shares a separate local plan.', 'all arms share the exact-origin SM86 SWA correction and retain original schedule selection.')
once('scope="Four consecutive TP8-shaped decoder blocks;', 'scope="Corrected SM86 SWA baseline; guarded18head prologue only. Four consecutive TP8-shaped decoder blocks;')
once('def plan(st,window):','def plan(st,window,local_lengths=None):')
once('cache_seqlens=st.l,qkv_dtype=op.b16,cu_seqlens_q=cuq,','cache_seqlens=st.l if local_lengths is None else local_lengths,qkv_dtype=op.b16,cu_seqlens_q=cuq,')
once('slide=plan(st,(left,0)) if cand else None',
 '''# Same exact-origin correction on every arm. No schedule optimization.
            origin=(st.l-n-left).clamp_min(0)
            local_lengths=st.l-origin
            columns=origin[:,None]+torch.arange(left+n,device=dev)[None,:]
            local_table=st.sw.gather(1,columns.long()).contiguous()
            slide=None''')
once('cache_seqlens=st.l,\n                    cache_batch_idx=',
 'cache_seqlens=st.l if block.full else local_lengths,\n                    cache_batch_idx=')
once('if block.full else st.sw,','if block.full else local_table,')
once('bank,side,stage,cand and m==128)','bank,side,stage,False)')
once('            for j,r in enumerate(outs["baseline"]):',
 '''            # Separate untimed backing-pool checks preserve timing symmetry.
            pool_reference=None
            for arm in names:
                poison();graphs[arm].replay();torch.cuda.synchronize()
                pools=[tensor for block in layers for tensor in (block.k8,block.v8,block.k,block.v)]
                if arm=="baseline":pool_reference=[x.view(torch.uint8).clone() for x in pools]
                else:
                    if not all(torch.equal(x.view(torch.uint8),y) for x,y in zip(pools,pool_reference)):
                        report["correct"]=False;cs["failure"]={"stage":"full_cache_backing","arm":arm,"round":i};save();return
            del pools,pool_reference
            cs["full_cache_checks"]=cs.get("full_cache_checks",0)+2
            cs["prologue_dispatch"]=[getattr(block,"prologue_adapter").last_reason for block in layers]
            assert all(reason=="enabled" for reason in cs["prologue_dispatch"])
            for j,r in enumerate(outs["baseline"]):''')
once('    report["complete"]=True;save()', '    report["complete"]=True;save()')
p=here/'chain_fixture.py';ast.parse(s);p.write_text(s)
(here/'chain_fixture_provenance.json').write_text(json.dumps(dict(parent_source=str(old),parent_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
 fixture_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),oracle=str(root/'research_2026-11-19/decode/attention_oracle_v2.json'),
 changed_operation='guarded18head prologue only; identical baseline metadata, MoE and corrected native SM86 attention on all arms',
 excludes='H200 FP8 attention, real peer sums, actual target weights, drafttransformer/fullserver'),indent=2))
print('prepared',p)
