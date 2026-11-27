"""Non-overlapping wall-clock stage measurements, with live progress."""
from contextlib import contextmanager
from collections import Counter
from time import perf_counter

class Timing:
    def __init__(self, enabled=True):
        self.enabled=enabled; self.totals=Counter(); self.current=None; self.records={}; self.page=''; self.started=perf_counter()
    def say(self, text):
        if self.enabled: print(text, flush=True)
    def begin(self, key, label):
        self.current=key; self.page=''; self.records[key]=Counter(); self.say('\n'+label)
    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield; return
        start=perf_counter()
        try: yield
        finally:
            elapsed=perf_counter()-start
            self.totals[name]+=elapsed
            if self.current is not None: self.records[self.current][name]+=elapsed
            self.say(f'  {self.page}{name:24s}: {elapsed:9.3f} s')
    def subtotal(self, key):
        self.say(f'  Measured PDF total      : {sum(self.records.get(key,{}).values()):9.3f} s')
    def summary(self):
        if not self.enabled: return
        self.say('\n--- Timing summary ---')
        for name, seconds in self.totals.most_common(): self.say(f'  {name:24s}: {seconds:9.3f} s')
        self.say(f'  Wall clock              : {perf_counter()-self.started:9.3f} s')
        self.say(f'  Mean measured PDF time  : {sum(sum(r.values()) for r in self.records.values())/max(1,len(self.records)):9.3f} s')

OFF=Timing(False)
