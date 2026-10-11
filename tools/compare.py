import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image, ImageDraw
from lutfit.loader import load_code, build_mask
from lutfit.tetra import evaluate

OUT='output'
x = np.load(f'{OUT}/lut33.npy')

def apply_lut(x, img, n=33, chunk=400_000):
    h,w,_ = img.shape
    flat = np.ascontiguousarray(img.reshape(-1,3), dtype=np.float64)
    out = np.empty_like(flat)
    for i in range(0,len(flat),chunk):
        out[i:i+chunk] = evaluate(x, flat[i:i+chunk], n)
    return np.clip(out,0,1).reshape(h,w,3)

def u8(im): return (np.clip(im,0,1)*255+0.5).astype(np.uint8)

def bar(w,text,h=24):
    b=Image.new('RGB',(w,h),(24,24,24)); ImageDraw.Draw(b).text((8,6),text,fill=(240,240,240)); return b

def panel(name, crop_frac=(0.0,0.0,1.0,1.0), colw=560):
    a = load_code(f'input/{name}', 1600)
    b = load_code(f'target/{name.lower()}', 1600)
    p = apply_lut(x, a)
    H,W,_ = a.shape
    x0,y0,x1,y1 = crop_frac
    sl = (slice(int(y0*H),int(y1*H)), slice(int(x0*W),int(x1*W)))
    tiles = [('A (input)',a[sl]), ('A + LUT33',p[sl]), ('B (target)',b[sl])]
    cols=[]
    for tx,arr in tiles:
        im=Image.fromarray(u8(arr))
        sc=colw/im.width
        im=im.resize((colw,max(1,int(im.height*sc))),Image.LANCZOS)
        cols.append((tx,im))
    hh=max(c.height for _,c in cols)+24
    canvas=Image.new('RGB',(colw*3,hh),(24,24,24))
    for i,(tx,im) in enumerate(cols):
        canvas.paste(bar(colw,tx),(i*colw,0))
        canvas.paste(im,(i*colw,24))
    return canvas

import sys as _s
jobs = [
  ('DSCF2163.JPG', (0.20,0.05,0.80,0.70), 'cmp_DSCF2163.png'),
  ('_DSC0321.jpg', (0.10,0.10,0.90,0.85), 'cmp_DSC0321.png'),
  ('DJI_20241003060901_0099_D.JPG', (0.05,0.05,0.95,0.80), 'cmp_DJI0099.png'),
  ('_DSC0346.jpg', (0.05,0.05,0.95,0.85), 'cmp_DSC0346.png'),
]
for name, crop, out in jobs:
    p = panel(name, crop)
    p.save(f'{OUT}/{out}')
    print('wrote', out, p.size)
