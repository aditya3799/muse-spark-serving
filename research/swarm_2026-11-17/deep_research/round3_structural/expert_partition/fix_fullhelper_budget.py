from pathlib import Path
p=Path(__file__).resolve().parent/'probe_swap_full_helper.py';s=p.read_text(encoding='utf-8-sig')
def one(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
one("entry_alloc=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()","entry_alloc=torch.cuda.memory_allocated();entry_reserved=torch.cuda.memory_reserved();torch.cuda.reset_peak_memory_stats()")
one("owned=torch.cuda.memory_reserved()-entry_alloc","owned=torch.cuda.memory_reserved()-entry_reserved;peak_owned=torch.cuda.max_memory_allocated()-entry_alloc")
one("'own_reserved_delta':owned,'peak_allocated'","'own_reserved_delta':owned,'own_peak_allocated_delta':peak_owned,'peak_allocated'")
one("assert free>=extra+reserve and owned+extra<=budget,('Memory budget',row)","assert free>=extra+reserve and owned+extra<=budget and peak_owned<=budget,('Memory budget',row)")
one("for expert in range(e):\n   ww,ss,gg=weight_fixture", "for expert in range(e):\n   memory('before_fixture_%s_%d'%(stage,expert),200*1024**2)\n   ww,ss,gg=weight_fixture")
one("with cached_threshold_for_preparation(a.out):baseline=ns['_prepare_humming_experts'](raw,torch)","gc.collect();torch.cuda.empty_cache();memory('before_prepare',300*1024**2)\n with cached_threshold_for_preparation(a.out):baseline=ns['_prepare_humming_experts'](raw,torch)")
one("canonical={stage:torch.empty_like", "memory('before_canonical_copy',256*1024**2)\n canonical={stage:torch.empty_like")
one("assert all(getattr(layers['B'],s+'_weight').data_ptr()==getattr(layers['C'],s+'_weight').data_ptr()!=getattr(baseline,s+'_weight').data_ptr()for s in canonical)","assert all(getattr(layers['B'],s+'_weight').data_ptr()==getattr(layers['C'],s+'_weight').data_ptr()!=getattr(baseline,s+'_weight').data_ptr()for s in canonical)\n weight_pointers={s:(getattr(baseline,s+'_weight').data_ptr(),getattr(baseline,s+'_weight_scale').data_ptr(),canonical[s].data_ptr())for s in canonical}")
one("for arm in 'ABC':graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))\n assert len({v[0].data_ptr()", "for arm in 'ABC':\n  memory('before_numeric_capture_'+arm,40*1024**2);graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))\n assert len({v[0].data_ptr()")
one("if generation:\n   for stage in canonical:","if generation:\n   memory('before_weight_scale_mutation',64*1024**2)\n   for stage in canonical:")
one("for pattern in ('uniform','hot','mixed','last','all_remote','tail_invalid'):","assert weight_pointers=={s:(getattr(baseline,s+'_weight').data_ptr(),getattr(baseline,s+'_weight_scale').data_ptr(),canonical[s].data_ptr())for s in canonical}\n  memory('after_generation_mutation')\n  for pattern in ('uniform','hot','mixed','last','all_remote','tail_invalid'):")
one("observed.clear();del graphs,outputs;gc.collect();torch.cuda.empty_cache();numeric=False", "observed.clear();del graph,v,graphs,outputs;gc.collect();torch.cuda.empty_cache();numeric=False")
one("for arm in 'ABC':graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))\n assert len({v.data_ptr()", "for arm in 'ABC':\n  memory('before_clean_capture_'+arm,20*1024**2);graphs[arm],outputs[arm]=capture(lambda arm=arm:full(arm))\n assert len({v.data_ptr()")
p.write_text(s,encoding='utf-8');compile(s,str(p),'exec');print('Updated peak budget, temporary/capture preflights, stable pointer asserts and numeric alias cleanup')
