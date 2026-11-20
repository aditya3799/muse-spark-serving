from pathlib import Path
import json
h=Path('/mnt/d/OpenKernelChallenge/serving/research_2026-11-19_round2/prefill')
for relative in ['combined_memory_ledger.json','early_projection_unapplied/phase_admission.json']:
 p=h/relative;j=json.loads(p.read_text());j['qualification_lifetime_correction']={'required_amendment':'row_no_grad_unapplied/incremental.diff SHA07f4c5184987eebcabefed4df902edaf1705a499575ad9f7a3959da54aed4387','frozen_factory_has_no_grad':False,'conditional_1703280640_bound':'Requires grad disabled or weight.requires_grad=False. Frozen factory does not guarantee either. CPU lifetime proof confirms reference may retain x.','unmodified_frozen_conservative_16k_bound_with_retention':1971716096};p.write_text(json.dumps(j,indent=2))
p=h/'early_projection_unapplied/REVIEW.txt'
s=p.read_text(encoding='utf-8-sig')
p.write_text('FOLLOW-UP: the1.703GB row qualification bound requires the separate no_grad amendment. Frozen530e factory lacks it; actual Torch CPU proof shows reference may retain x, raising conservative16K transport bound to1.9717GB. See ../row_no_grad_unapplied and autograd_lifetime.json. Earlier relocation alone does not repair this.\n\n'+s)
