"""Exercise the GUI without a screen: build it, feed it a result, dump what it shows."""
import os, sys, pickle, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import app as appmod
from lutfit.pipeline import apply_lut, with_strength

with open('work/gui_demo.pkl','rb') as f: res = pickle.load(f)

a = appmod.App()
a.var_in.set(os.path.abspath('input')); a.var_out.set(os.path.abspath('target'))
a.update_idletasks(); a.update()
a._finish(res)
for _ in range(5): a.update(); time.sleep(0.05)

print("=" * 70); print("预检 TAB"); print("=" * 70)
print(a.txt_pre.get("1.0","end").rstrip())
print(); print("=" * 70); print("报告 TAB"); print("=" * 70)
print(a.txt_rep.get("1.0","end").rstrip())

# reproduce exactly what the preview canvas draws
n = res.options.lut_size
x = with_strength(res.lut, n, 1.0)
cols = []
for src in a._preview_src:
    pred = apply_lut(x, src['a'], n)
    cols.append(np.concatenate([src['a'], pred, src['b']], axis=0))
comp = np.concatenate(cols, axis=1)
Image.fromarray((np.clip(comp,0,1)*255+0.5).astype(np.uint8)).save('output/gui_canvas.png')
print("\npreview canvas ->", comp.shape, "saved output/gui_canvas.png")

# exercise the strength slider path
for s in (0.35, 0.0, 1.0):
    a.strength.set(s); a._on_strength(); a.update()
    print(f"  strength {s:.2f} -> label {a.var_strength.get()!r}  ok")
print("\nGUI exercised with no exceptions.")
a.destroy()
