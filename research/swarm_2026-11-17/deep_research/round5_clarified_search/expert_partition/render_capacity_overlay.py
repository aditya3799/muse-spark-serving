import pathlib,ast,difflib,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=next(p for p in P.parents if (p/'RULES.md').exists())
SOURCE=R/'swarm_2026-11-16/coordination/round5_humming_adaptive_fused/candidate_sources/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'
text=SOURCE.read_text(encoding='utf8')
changes=[
('def choose_partition_configs(layer, actual_routes, global_experts):\n    experts = layer.w13_post_scale.numel()',
 'def choose_partition_configs(layer, actual_routes, global_experts, policy_experts=None):\n    experts = layer.w13_post_scale.numel() if policy_experts is None else policy_experts'),
('    valid = local_ids.numel()\n    experts = layer.w13_post_scale.numel()\n',
 '    valid = local_ids.numel()\n    experts = layer.w13_post_scale.numel()\n    # Opt-in research metadata separates allocated slots from baseline policy.\n    policy_experts = getattr(layer, "_research_policy_local_experts", experts)\n'),
('            or global_experts < experts or global_experts % experts\n',
 '            or not isinstance(policy_experts,int) or isinstance(policy_experts,bool)\n            or not 0 < policy_experts <= experts\n            or global_experts < experts or global_experts % policy_experts\n'),
('    plan = choose_partition_configs(layer,valid,global_experts)\n',
 '    plan = choose_partition_configs(layer,valid,global_experts,policy_experts=policy_experts)\n')]
out=text
for old,new in changes:
 assert out.count(old)==1,old
 out=out.replace(old,new)
ast.parse(out)
(P/'humming_split_capacity_overlay.py').write_text(out,encoding='utf8')
diff=''.join(difflib.unified_diff(text.splitlines(True),out.splitlines(True),fromfile='a/python/sglang/kernels/ops/moe/muse_spark_humming_split.py',tofile='b/python/sglang/kernels/ops/moe/muse_spark_humming_split.py'))
(P/'humming_split_capacity_overlay.diff').write_text(diff,encoding='utf8')
(P/'CAPACITY_OVERLAY_MANIFEST.json').write_text(json.dumps({'status':'DEFAULT_OFF_LOGICAL_PHYSICAL_SEPARATION_SOURCE_ONLY','source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'overlay_sha256':hashlib.sha256((P/'humming_split_capacity_overlay.py').read_bytes()).hexdigest(),'scope':'Absent _research_policy_local_experts selects original physical count and same original decisions. Physical count still controls histogram, alignment, slot validity and postscale cardinality. Opt-in policy32 preserves expected-local selection for physical capacities33/40/64.','requirements':['Actual Humming tuning tables/configs must be pinned against original baseline, not assumed unchanged by capacity.','Physical packed weights/scales and metadata must already cover every resident slot.','Global logical expert IDs and global_experts256 remain unchanged.','This does not install or qualify replica weight publication or GPU execution.']},indent=2)+'\n')
print('Capacity overlay rendered.')
