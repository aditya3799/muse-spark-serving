import hashlib,json,pathlib
P=pathlib.Path(__file__).resolve().parent
v=(P/'virtual_attention_ar.cu').read_text(encoding='utf-8-sig');assert v.count('kPostWeight=true')==3
v=v.replace('kPostWeight=true','kPostWeight=false')
(P/'virtual_attention_actual.cu').write_bytes(v.encode())
h=(P/'probe_virtual_attention.py').read_text(encoding='utf-8-sig').replace('virtual_attention_ar.cu','virtual_attention_actual.cu').replace('probe_virtual_attention.py','probe_virtual_attention_actual.py').replace('weight.data_ptr(),pre.data_ptr()','0,pre.data_ptr()').replace('1e-6,1e-6,m','1e-8,1e-5,m').replace('alpha.uniform_(.2,.8);beta.uniform_(.2,.8)','gate=torch.randn_like(alpha)/.3;beta.copy_(torch.sigmoid(gate));alpha.copy_(torch.sqrt(torch.clamp(torch.sigmoid(-gate)*(1+beta),min=1e-3)))')
(P/'probe_virtual_attention_actual.py').write_bytes(h.encode())
mc=(P/'offline_mc_ready.cu').read_text(encoding='utf-8-sig');assert mc.count('kHasRouter=true,kPostWeight=true')==2
mc=mc.replace('kHasRouter=true,kPostWeight=true','kHasRouter=false,kPostWeight=false')
(P/'offline_mc_actual.cu').write_bytes(mc.encode())
report=dict(status='TARGET_BRANCH_SOURCE_QUALIFICATION_PENDING',frozen_model='round5_humming_adaptive_fused/candidate_sources/python/sglang/srt/models/muse_spark.py',configuration_defaults=dict(H=8192,attention_heads=128,head_dim=64,localK_TP8=1024,post_norm_eps=1e-8,pre_norm_eps=1e-5,channel_wise_gate=True,gate_temperature=.3,gate_alpha_epsilon=1e-3,post_weight=False,pre_weight=True,zero_centered_pre_weight=True,pre_norm_bf16=False),shape_specializations=[dict(M='1..8',router=True,threads=256,prefetch=True,communication='one-shot-push'),dict(M='32..191',router=False,threads=256,prefetch=True,communication='MC2shot when selected'),dict(M='192..1024',router=False,threads=512,prefetch=False,communication='MC2shot when selected')],checkpoint_caveat='eps/gate defaults source-grounded; futuremodel integration must readactual modulevalues rather than hardcode',general_fixture='virtual_ar_02 had post_weight=True/router=True andeps1e-6; retainas generalmath result, notactualbranch/resource qualification',sources={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['virtual_attention_actual.cu','probe_virtual_attention_actual.py','offline_mc_actual.cu']})
(P/'ACTUAL_ATTENTION_BRANCH_CONTRACT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

