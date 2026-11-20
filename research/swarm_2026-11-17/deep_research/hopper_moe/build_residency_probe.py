"""Prepare the local analogous residency gate from the successful grid fixture."""
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'grid_probe.py').read_text()
begin=s.index('    def make_view(grid):')
end=s.index('    flush=torch.empty',begin)
s=s[:begin]+'''    from residency_view import residency_layer_view
    capability=torch.cuda.get_device_capability()
    sm_version=capability[0]*10+capability[1]
    if sm_version!=86:raise RuntimeError('This pilot is explicitly SM86; Hopper has a separate gate')
    views={}
    view_reports={}
    for stages,ctas in ((2,1),(2,2),(3,2)):
      label='s%d_cta%d'%(stages,ctas)
      views[label],view_reports[label]=residency_layer_view(humm,stages,ctas,sm_version)
    report['view_reports']=view_reports
''' + s[end:]
s=s.replace("'scope':'Same compiled GEMM and arithmetic; only persistent grid size changes. SM86 component evidence, not H200 or serving.'", "'scope':'Same block/warp/K and arithmetic, changed pipeline-depth and residency. SM86 numerical/component gate, not H200 or serving.'")
s=s.replace("item={'grid_sms':int(label),'numerical':numerical}", "item={'policy':label,'numerical':numerical}")
(P/'residency_probe_sm86.py').write_text(s)
print('SM86 residency probe prepared')
