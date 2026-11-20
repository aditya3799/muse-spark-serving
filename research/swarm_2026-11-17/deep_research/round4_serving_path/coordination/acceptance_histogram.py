"""CPU-only observer of SGLang's existing completed-verify callback.
Not installed in active serving. No tensor imports, events or extra D2H work.
Fixed8 DFlash baseline only; describes pre-grammar/pre-termination commitments.
"""
import functools
import threading

class AcceptanceHistogram:
    def __init__(self, *, every=1, max_steps=8192, max_batch=128):
        if min(every,max_steps,max_batch)<1:raise ValueError('positive limits required')
        self.every=every;self.max_steps=max_steps;self.max_batch=max_batch
        self.calls=0;self.sampled=0;self.invalid=0;self.rows={};self._undo=None
        self.thread=threading.get_ident()
    def record(self, correct_drafts, batch_size):
        self.calls+=1
        if (self.calls-1)%self.every or self.sampled>=self.max_steps:return
        if (threading.get_ident()!=self.thread or type(batch_size) is not int
            or not 1<=batch_size<=self.max_batch or type(correct_drafts) is not list
            or len(correct_drafts)!=batch_size
            or any(type(v) is not int or not 0<=v<=7 for v in correct_drafts)):
            self.invalid+=1;return
        row=self.rows.setdefault(batch_size,{'steps':0,'correct_draft_histogram':[0]*8,
            'survivor_count_histograms':{w:[0]*(batch_size+1) for w in (2,4,6)}})
        row['steps']+=1;self.sampled+=1
        for value in correct_drafts:row['correct_draft_histogram'][value]+=1
        for width,counts in row['survivor_count_histograms'].items():
            counts[sum(value>=width for value in correct_drafts)]+=1
    def install(self, worker):
        if self._undo is not None:raise RuntimeError('already installed')
        if type(worker).__name__!='DFlashWorkerV2' or int(worker.block_size)!=8:
            raise ValueError('only fixed8 DFlash worker admitted')
        old=worker.on_verify_complete_cpu
        if not callable(old):raise TypeError('CPU callback missing')
        had='on_verify_complete_cpu' in vars(worker);previous=vars(worker).get('on_verify_complete_cpu')
        @functools.wraps(old)
        def observe(num_correct_drafts_per_req,batch_size=0):
            result=old(num_correct_drafts_per_req,batch_size=batch_size)
            if int(worker.block_size)==8:self.record(num_correct_drafts_per_req,batch_size)
            else:self.invalid+=1
            return result
        worker.on_verify_complete_cpu=observe
        self._undo=(worker,had,previous,observe)
        return self
    def uninstall(self):
        if self._undo is None:return
        worker,had,previous,observe=self._undo
        if worker.on_verify_complete_cpu is not observe:raise RuntimeError('callback ownership changed')
        if had:worker.on_verify_complete_cpu=previous
        else:delattr(worker,'on_verify_complete_cpu')
        self._undo=None
    def snapshot(self):
        return {'scope':'Fixed8 DFlash completed CPU callback; counts before grammar/termination/retraction filtering',
            'sample_every':self.every,'calls':self.calls,'sampled_steps':self.sampled,'invalid_samples':self.invalid,
            'by_batch_size':self.rows,
            'interpretation':'L=correct_drafts+1; continuation after widthW requires correct_drafts>=W. Joint survivor histograms retain per-batch correlation.',
            'limitations':['Descriptive lagged counts, not a live rank-coherent controller','No phase timing or new GPU work; CPU observer overhead not benchmarked',
                           'Do not mix different verification policies in one histogram','No prompts, token IDs, request IDs or per-request histories stored']}
