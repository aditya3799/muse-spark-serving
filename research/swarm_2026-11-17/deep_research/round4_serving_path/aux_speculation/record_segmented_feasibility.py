import hashlib,json
from pathlib import Path
P=Path(__file__).parent;ROOT=P.parents[3];SRT=ROOT/'sglang/python/sglang/srt'
sources=['speculative/dflash_worker_v2.py','speculative/dflash_info.py','mem_cache/kv_loc_plan.py','layers/attention/flashattention_backend.py','model_executor/runner/decode_cuda_graph_runner.py','model_executor/model_runner_components/weight_updater.py','managers/scheduler.py','managers/scheduler_components/metrics_reporter.py','managers/tokenizer_manager.py']
r={'status':'SOURCE_FEASIBLE_NUMERIC_AND_COST_HELD','proposal':'Segment the SAME fixed8 draft proposals, first4targetinputs then4tailinputs for survivors; no redraft',
'contrast_with_capped_prefix':'Capped prefix emits an earlier target bonus and redrafts next iteration; segmented mode retains full8 proposal and baseline acceptance length only if segmented model/aux computations agree.',
'algorithm':[
 'Keep full8 proposal IDs and original full8 target write window/reservation. No scheduler publication at intermediate stage.',
 'Stage1 target inputs c0..c3 at positions L..L+3. Compare predictions p0..p3 to c1..c4, including the extra continuation edge p3==c4.',
 'If first mismatch is j in0..3, final accepted=a=j, bonus=pj, commit=a+1. No tail target rows needed for that request.',
 'If all4edges match, the request survives with >=4correct proposals. Stage2 inputs c4..c7 at positions L+4..L+7 and context prefixL+4; original targetKV first4rows already exists.',
 'Stable or explicitly mapped compaction gathers original request indices, lengths, candidate rows, positions and full/SWA write slots together. A new KVLocPlan is required; reusing the original unfiltered plan has wrong row ownership/prefix.',
 'Consume/copy firstaux BEFORE any second target forward overwrites shared aux output. Prefix-valid draftKV stage1commit=min(finalcommit,4), which is4 for every survivor and exact for non-survivors.',
 'Tail stage comparisons p4..p6 vs c5..c7 yield finala=4+tailaccepted. Tail draftKVcommit=a+1-4. No bonus token is written toKV untilnextiteration.',
 'Merge results back to original request order, retain outputstride8/draftproposalaccount8 and publish finalsequence lengths ONCE; then nextdraft uses original full8-step result.'
],
'source_obligations':{
 'accept_boundary':'dflash_worker_v2._accept_block assumes candidateswidth == block_size. OrdinaryW4 only tests3edges, so segmented continuation needs a separate fourth-edge decision; do not reuse cappedW4 result blindly.',
 'explicit_slots':'_append_target_hidden_to_draft_kv_by_loc accepts cache_loc_2d/commit_lens independent of globalblockwidth. It can represent two writes, but split projection shape changes are not numerically qualified.',
 'aux_lifetime':'forward_batch_generation near2975 explicitly states every decode graph size shares one aux output, overwritten by nexttargetforward. Stagingcost or earlyappend is mandatory.',
 'target_graphs':'Dedicatedwidth4 targetrunner can serve bothstages, with differentrequestcounts/seq_lens and stablecapturedbuffers. Realdecoderunner/attention stillunqualified; metadata-only fixture is notthisfullpath.',
 'rank_control':'S and survivororder determine targetcollective shapes/order. Must be identical acrossTP/EP ranks. Local acceptance/timing decisions cannot independently diverge.',
 'host_sync':'Selecting host-side graphbucket from dynamicGPU survivorcount needs a completion/hosttransfer or alternativedevice-launch mechanism. Existing end-of-step hostresults do not eliminate this new mid-step dependency.',
 'modes':'Initialscope plain greedy, topk1, serial fullTP, noDP/CP/PP/TBO/PDMux/Mamba/LoRA/LiLiCorr/Domino, no grammar/logprob/customlogits; originalfallback retained.',
 'mutations':'Prefix runtime invalidation must run BEFORE any weightwrite/postprocess/recapture. WeightUpdater.begin_weight_update is notuniversal: disk path restores/loads directly; load_weights_from_distributed likewise loads directly. One end-of-update hook is insufficient.'},
'cost_criterion':{
 'baseline':'D8(B,L)+T8(B,L)+A8(B,L)+existing scheduling',
 'segmented':'D8(B,L)+T4(B,L)+I(S>0)*T4(S,L+4)+Afirst4+Atail4(S)+decision+compaction+extra scheduling/publication cost',
 'required_measurement':'Measured actual fullstage durations conditioned on B,L,S, routes and graphbucket. Use clean pairedfullpath; do not scaleT8 linearly bywidth.',
 'highload_counterpoint':'AtTP8/EP8, B32 width8→4 gives tokenrows256→128. Uniformroutes may stillvisitall32localexperts withsame lighttile, readingallweights; twoforwards mayreadthem twice.',
 'fixed_capacity_fallback':'Always replaying tail atB withmasked nonsurvivors avoids a hostSdecision but does notestablish workreduction. Generic layer kernels may stillprocesspaddedrows andreadweights.',
 'aux_tradeoff':'Earlyappend eliminates firstauxcopy but splits expensive replicated40960→8192 and draftKV projections; savedtailrows must outweigh secondlaunch/GEMMshape costs. Staging firstaux insteadadds explicitcopy and~B*4*40960*2 bytes (10MiB atB32).',
 'kill_criterion':'Reject if measuredT4(B)+T4(S)+incrementaldecision/aux >=T8+A8 at observedsurvivor distribution, or iffullmodelgreedy/KVagreement fails. No claimed gain withoutthese data.',
 'log_limit':'Saved acceptance4.0–4.1 is aggregate, notsurvivorhistogram or perbatchcorrelation. Meancommit4.1 cancoexistwith survival0.1 or0.5 (CPUcounterexample); neither is inferredforMuse.'},
'primary_context':{'url':'https://arxiv.org/html/2605.00342','title':'Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding','accessed_utc':'2026-11-17','note':'Prior work already combines verification-prefix selection with precaptured width-specific targetgraphs and profiled costs. This research makes no novelty claim and transfers no published speedup to Muse. Segmented sameproposal continuation is a separate local hypothesis.'},
'next_decisive_test':'After explicit authorization and suitable hardware: fixedproposal targetW8 vs segmentedW4+W4 withrealFP8KV, fullaux/draftKV equality, routes/collectives and allcosts. Localmetadata and idealoracle tests alone cannotpromote.',
'sources':{str(p):hashlib.sha256((SRT/p).read_bytes()).hexdigest() for p in sources}}
(P/'SEGMENTED_VERIFY_FEASIBILITY.json').write_text(json.dumps(r,indent=2)+'\n')
print('saved segmented feasibility')