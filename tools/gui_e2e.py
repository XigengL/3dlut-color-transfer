"""Drive the GUI exactly as a user would: set folders, click 开始拟合, wait."""
import os, sys, time, dataclasses
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as appmod
from lutfit.pipeline import FitOptions

# keep the test quick, but exercise the identical code path
Base = appmod.FitOptions
class Small(Base):
    def __init__(self, **kw):
        kw.setdefault('max_pairs', 4)
        kw.setdefault('grid', 33)
        kw.setdefault('target_max', 700)
        kw.setdefault('eval_max', 600)
        kw.setdefault('preview_count', 2)
        super().__init__(**kw)
        self.grid = kw['grid']; self.target_max = 700
        self.eval_max = 600; self.preview_count = 2
    def scaled(self): return self
appmod.FitOptions = Small

a = appmod.App()
a.var_in.set(os.path.abspath('input')); a.var_out.set(os.path.abspath('target'))
a.update_idletasks(); a.update()

print("clicking 开始拟合 ...")
a._start()
seen = set(); t0 = time.time()
while a.worker is not None and a.worker.is_alive() or not a.q.empty():
    a.update()
    msg = a.var_status.get()
    seen.add(msg.split('   ')[-1][:24])
    time.sleep(0.05)
    if time.time() - t0 > 600: print("TIMEOUT"); break
for _ in range(20): a.update(); time.sleep(0.05)

print(f"finished in {time.time()-t0:.0f}s")
print("status:", a.var_status.get())
print("result present:", a.result is not None)
print("export button:", a.btn_export['state'])
print("slider:", a.scale['state'])
print("preview columns:", len(a._preview_src))
print("preflight tab enabled:", a.nb.tab(1,'state'))
print("report tab enabled:", a.nb.tab(2,'state'))
print("distinct progress messages seen:", len(seen))
for s in sorted(seen)[:6]: print("   ", s)

# cancel path
a2 = appmod.App()
a2.var_in.set(os.path.abspath('input')); a2.var_out.set(os.path.abspath('target'))
a2.update(); a2._start(); time.sleep(0.4); a2._cancel()
t0=time.time()
while (a2.worker is not None and a2.worker.is_alive() or not a2.q.empty()) and time.time()-t0 < 30:
    a2.update(); time.sleep(0.05)
for _ in range(10): a2.update(); time.sleep(0.05)
print("cancel ->", a2.var_status.get(), "| go button:", a2.btn_go['state'])
a.destroy(); a2.destroy()
print("\nE2E OK, no exceptions")
