"""Independent source-only review of coordinator attention composition."""
import ast,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
from pathlib import Path
from types import SimpleNamespace as NS
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
COORD=ROOT/'coordination/research_resumed'
REL='python/sglang/srt/layers/attention/'
BACK=REL+'flashattention_backend.py';HELP=REL+'fa3_swa_scheduler.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
conservative=COORD/'norm_glue_router/candidate_sources'
optional=COORD/'norm_glue_router_swa/candidate_sources'
ref=P/'glue_swa_snapshot/candidate_sources'
glue=P.parent/'glue_only_ablation/candidate_sources'
assert sha(conservative/BACK)==sha(glue/BACK)
assert sha(optional/BACK)==sha(ref/BACK)
assert sha(optional/HELP)==sha(ref/HELP),'Recompose LF-normalized helper before final review'
assert ast.dump(ast.parse((optional/HELP).read_bytes()))==ast.dump(ast.parse((P/'candidate_sources'/HELP).read_bytes()))
files=lambda root:{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
a,b=files(conservative),files(optional)
assert set(b)-set(a)=={HELP} and not set(a)-set(b)
assert {key for key in a if a[key]!=b[key]}=={BACK}
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
cpu=load(P/'test_lifecycle.py','composed_contract_test')
cpu.P=ROOT;cpu.HELPER=optional/HELP;cpu.BACKEND=optional/BACK;cpu.h=load(optional/HELP,'composed_swa_helper')
# Existing exact-source CPU lifecycle/guards against composed backend and helper.
result=cpu.run()
def get_method(path,name):
 cls=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
 return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
combined_tree=ast.parse((optional/BACK).read_text(encoding='utf-8'))
combined_class=next(n for n in combined_tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttentionBackend')
combined_class.body=[n for n in combined_class.body if not(isinstance(n,ast.FunctionDef) and n.name=='glue_draft_metadata')]
assert ast.dump(combined_tree)==ast.dump(ast.parse((P/'candidate_sources'/BACK).read_text(encoding='utf-8')))
glue_node=get_method(optional/BACK,'glue_draft_metadata')
assert ast.dump(glue_node)==ast.dump(get_method(glue/BACK,'glue_draft_metadata'))
globals_={'SpeculativeAlgorithm':NS(from_string=lambda value:NS(is_dflash=lambda:True)), 'get_spec':lambda:NS(speculative_algorithm='DFLASH'),'get_parallel':lambda:NS(attn_cp_size=1),'_should_disable_scheduler_metadata_precompute':lambda:False}
exec(compile(ast.Module(body=[glue_node],type_ignores=[]),'<exact_composed_glue>','exec'),globals_)
rows=[]
for fa in (3,4):
 for draft in (False,True):
  for verify in (False,True):
   factory,backend,fb,meta,layer,kwargs=cpu.fixture()
   backend.fa_impl_ver=fa;backend.is_draft_runner=draft
   backend.speculative_num_draft_tokens=8;backend.is_encoder_decoder=False
   backend.use_sliding_window_kv_pool=False;backend.needs_cpu_seq_lens=False
   backend.kv_index_translator=NS(is_translating=False,reads_are_translated=False)
   fb.forward_mode.is_target_verify=lambda:verify
   why=globals_['glue_draft_metadata'](backend)
   cpu.h.reset_swa_scheduler_cache(backend,fb,allow=True)
   plan=cpu.h.get_swa_verify_scheduler_metadata(**kwargs)
   assert (why=='')==(fa==4 and draft)
   assert (plan is not None)==(fa==3 and verify)
   assert not(why=='' and plan is not None)
   rows.append(dict(fa=fa,draft_runner=draft,target_verify=verify,glue_eligible=why=='',swa_scheduler_reused=plan is not None))
# Same backend crossing mode boundaries clears prior per-forward scheduler.
factory,backend,fb,meta,layer,kwargs=cpu.fixture()
for verify in (True,False,True,False):
 fb.forward_mode.is_target_verify=lambda:verify
 cpu.h.reset_swa_scheduler_cache(backend,fb,allow=True)
 plan=cpu.h.get_swa_verify_scheduler_metadata(**kwargs)
 assert (plan is not None)==verify
assert len(factory.calls)==2
repository=ROOT.parent/'sglang/python/sglang/srt'
runner_files=[repository/'model_executor/runner/decode_cuda_graph_runner.py',repository/'model_executor/runner/metadata_glue_graph.py',repository/'model_executor/forward_batch_info.py']
report=dict(status='INDEPENDENT_COMPOSED_ATTENTION_SOURCE_REVIEW_PASS',source_hashes={'conservative_attention':sha(conservative/BACK),'optional_attention':sha(optional/BACK),'optional_helper':sha(optional/HELP),'native_tested_helper':sha(P/'candidate_sources'/HELP)},conservative_matches_glue_only=True,optional_matches_glue_swa_export=True,all_other_composed_files_identical=True,source_count={'conservative':len(a),'optional':len(b)},export_helper_ast_matches_native_tested_helper=True,cpu_lifecycle=result,coexistence_matrix=rows,mode_transition_checks=4,source_notes=[
 'Glue eligibility requires draft runner and FA4 DFlash fixed-width metadata. SWA scheduler dispatch requires FA3 and TARGET_VERIFY; neither mechanism can consume the other plan.',
 'DRAFT_EXTEND_V2 and TARGET_VERIFY are distinct ForwardMode members. A draft runner may also use TARGET_VERIFY capture mode; FA version guards still separate the two optimizations.',
 'Combined in_graph reset occurs before existing draft-extend early branches; non-verify draft mode sets scheduler cacheNone. Eager and out_graph resets remain in place.',
 'MetadataGlueGraph captures only out_graph prep and reinstalls forward_metadata, not backend scheduler dictionary. Model capture runs in_graph reset before layers. Replay executes captured native scheduler operations without consulting Python dictionary; subsequent eager/capture entry resets before dispatch.',
 'Existing glue method AST is unchanged; removing it from optional backend restores standalone SWA AST. Norm/router composition adds no additional attention source changes.'
 ],runner_source_sha256={str(f.relative_to(ROOT.parent)):sha(f) for f in runner_files},limitations=['Source/CPU coexistence review only; composed full model never executed here','Native SM86 BF16 isolated helper evidence does not validate FA4 draft glue or H200 FP8','Same-stream sequential execution assumed; TBO/pdmux/compile/CP exclusions preserved'],gpu_or_compiler_launched=False)
# Avoid absolute-location issue in reused report source keys; result is still exact.
(P/'COMPOSED_ATTENTION_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'receipt_sha256':sha(P/'COMPOSED_ATTENTION_REVIEW.json'),'matrix_cases':len(rows),'sources':report['source_hashes']}))
