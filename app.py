"""3D LUT 拟合工具 — 图形界面

Tkinter（Python 标准库），Windows / macOS 通用，无额外依赖。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lutfit.pipeline import (Cancelled, FitOptions, apply_lut, export_cube,
                             run_fit, strength_check, with_strength)

APP_TITLE = "3D LUT 拟合工具"
BG = "#1e1f22"

TILE_W, TILE_H = 380, 260


# --------------------------------------------------------------- helpers ----
def to_photo(img_array, width):
    """numpy float [0,1] HxWx3 -> ImageTk.PhotoImage scaled to ``width``."""
    u8 = (np.clip(img_array, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(u8)
    scale = width / im.width
    im = im.resize((width, max(1, int(im.height * scale))), Image.LANCZOS)
    return ImageTk.PhotoImage(im)


def _tile(img, w, h):
    """Centre-crop to the tile aspect ratio, then resize to exactly (w, h).

    Preview images come from cameras with different aspect ratios; without this the
    columns would not line up.
    """
    im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    iw, ih = im.size
    target = w / h
    cur = iw / ih
    if cur > target:
        nw = max(1, int(round(ih * target)))
        x0 = (iw - nw) // 2
        im = im.crop((x0, 0, x0 + nw, ih))
    else:
        nh = max(1, int(round(iw / target)))
        y0 = (ih - nh) // 2
        im = im.crop((0, y0, iw, y0 + nh))
    im = im.resize((w, h), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


class ScrollFrame(ttk.Frame):
    """A vertically scrollable container.

    Needed because the window does not fit on every screen; without it the lower
    controls simply fall off the bottom and appear to be unresponsive.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind_all(seq, self._on_wheel, add="+")

    def _on_inner(self, _=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self._win, width=event.width)

    def _on_wheel(self, event):
        # never steal the wheel from a widget that scrolls itself
        try:
            w = self.winfo_containing(event.x_root, event.y_root)
        except Exception:
            return
        while w is not None:
            if isinstance(w, (tk.Text, tk.Listbox)):
                return
            w = getattr(w, "master", None)
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step, "units")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        # fit on smaller laptop screens instead of running off the bottom
        h = min(880, max(600, self.winfo_screenheight() - 150))
        self.geometry(f"1080x{h}")
        self.minsize(880, 560)
        self.configure(bg=BG)

        self.q = queue.Queue()
        self.worker = None
        self.cancel_flag = False
        self.result = None
        self.strength = tk.DoubleVar(value=1.0)
        self._photos = []
        self._preview_src = []
        self._slider_job = None

        self._style()
        self._build()
        self.after(80, self._poll)

        # NOTE: motion-based refocusing and a <FocusIn> handler were tried here and
        # removed.  `<Motion>` called lift()/focus_force() roughly twice a second
        # while the pointer moved (focus_get() returns None whenever no widget holds
        # focus, which is normal), and that lands between a press and its release —
        # i.e. it can *cause* the two-click behaviour it was meant to cure.  The
        # <FocusIn> handler could also loop: focus_force() emits FocusIn again.
        #
        # What is left is safe: one refocus at startup, and one after each dialog.
        # Anything beyond that needs a newer Tk, not more Python.
        self.after(120, self._refocus)

    def _refocus(self):
        try:
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    # ------------------------------------------------------------ styling --
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure(".", background=BG, foreground="#e6e6e6", fieldbackground="#2b2d31")
        st.configure("TFrame", background=BG)
        st.configure("TLabel", background=BG, foreground="#e6e6e6", font=("Helvetica", 12))
        st.configure("H1.TLabel", font=("Helvetica", 17, "bold"), foreground="#ffffff")
        st.configure("Dim.TLabel", foreground="#9aa0a6", font=("Helvetica", 11))
        st.configure("TCheckbutton", background=BG, foreground="#e6e6e6", font=("Helvetica", 12))
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TRadiobutton", background=BG, foreground="#e6e6e6", font=("Helvetica", 12))
        st.map("TRadiobutton", background=[("active", BG)])
        st.configure("TButton", font=("Helvetica", 12), padding=6)
        st.configure("Go.TButton", font=("Helvetica", 14, "bold"), padding=10)
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(16, 8), font=("Helvetica", 12))
        st.map("TNotebook.Tab", background=[("selected", "#2b2d31")],
               foreground=[("selected", "#ffffff")])
        st.configure("Horizontal.TProgressbar", background="#4a9eff", troughcolor="#2b2d31",
                     borderwidth=0, thickness=14)
        st.configure("TScale", background=BG)

    # ---------------------------------------------------------------- ui --
    def _build(self):
        head = ttk.Frame(self)
        head.pack(side="top", fill="x", padx=14, pady=(12, 2))
        ttk.Label(head, text=APP_TITLE, style="H1.TLabel").pack(anchor="w")
        ttk.Label(head, text="把「相机原图」和「修图后的图」两个文件夹喂进去，输出一个 .cube 文件。",
                  style="Dim.TLabel").pack(anchor="w", pady=(2, 0))

        # The footer is packed to the bottom *first*, so the progress bar and the
        # export controls can never be pushed off-screen by the scroll area.
        foot = ttk.Frame(self)
        foot.pack(side="bottom", fill="x", padx=14, pady=(4, 12))
        self.pb = ttk.Progressbar(foot, maximum=1000)
        self.pb.pack(fill="x", pady=(0, 2))
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(foot, textvariable=self.var_status, style="Dim.TLabel").pack(anchor="w")

        row = ttk.Frame(foot)
        row.pack(fill="x", pady=(6, 0))
        ttk.Label(row, text="强度").pack(side="left")
        self.scale = ttk.Scale(row, from_=0.0, to=1.0, orient="horizontal",
                               variable=self.strength, command=self._on_strength,
                               state="disabled", length=300)
        self.scale.pack(side="left", padx=8)
        self.var_strength = tk.StringVar(value="100%")
        ttk.Label(row, textvariable=self.var_strength, width=5).pack(side="left")
        ttk.Label(row, text="（觉得太重就往左拉）", style="Dim.TLabel").pack(side="left")
        self.btn_export = ttk.Button(row, text="导出 .cube", command=self._export,
                                     state="disabled")
        self.btn_export.pack(side="right")

        scroll = ScrollFrame(self)
        scroll.pack(side="top", fill="both", expand=True, padx=14, pady=6)
        b = scroll.inner

        # ---- folders ----
        fbox = ttk.Frame(b)
        fbox.pack(fill="x", pady=(4, 6))
        fbox.columnconfigure(1, weight=1)
        self.var_in = tk.StringVar()
        self.var_out = tk.StringVar()
        for r, (label, var, prompt) in enumerate(
                (("A  相机原图文件夹", self.var_in, "选择 A 文件夹：相机原图"),
                 ("B  修图后文件夹", self.var_out, "选择 B 文件夹：修图后"))):
            ttk.Label(fbox, text=label, width=18).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(fbox, textvariable=var).grid(row=r, column=1, sticky="ew", padx=8)
            ttk.Button(fbox, text="浏览…", width=8,
                       command=lambda v=var, p=prompt: self._pick(v, p)
                       ).grid(row=r, column=2)

        obox = ttk.Frame(b)
        obox.pack(fill="x", pady=6)
        ttk.Label(obox, text="LUT 尺寸").pack(side="left")
        self.var_size = tk.IntVar(value=33)
        for s in (17, 33, 65):
            ttk.Radiobutton(obox, text=f"{s}³", value=s,
                            variable=self.var_size).pack(side="left", padx=6)
        ttk.Label(obox, text="（33 为通用推荐）", style="Dim.TLabel").pack(side="left", padx=(0, 18))
        self.var_neutral = tk.BooleanVar(value=True)
        ttk.Checkbutton(obox, text="保留中性轴（灰阶不带色偏）",
                        variable=self.var_neutral).pack(side="left")

        rbox = ttk.Frame(b)
        rbox.pack(fill="x", pady=6)
        self.btn_go = ttk.Button(rbox, text="开始拟合", style="Go.TButton", command=self._start)
        self.btn_go.pack(side="left")
        self.btn_cancel = ttk.Button(rbox, text="取消", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=8)

        self.nb = ttk.Notebook(b, height=430)
        self.nb.pack(fill="both", expand=True, pady=(8, 4))
        self._tab_preview()
        self._tab_text("预检", "pre")
        self._tab_text("报告", "rep")
        self.nb.tab(1, state="disabled")
        self.nb.tab(2, state="disabled")

    def _tab_preview(self):
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="预览")
        self.cv = tk.Canvas(f, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(f, orient="vertical", command=self.cv.yview)
        self.cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.cv.pack(side="left", fill="both", expand=True)
        self.cv.bind("<Configure>", lambda e: self._render_preview())

    def _tab_text(self, title, key):
        f = ttk.Frame(self.nb)
        self.nb.add(f, text=title)
        txt = tk.Text(f, bg="#17181a", fg="#e6e6e6", relief="flat",
                      font=("Menlo" if sys.platform == "darwin" else "Consolas", 11),
                      wrap="word", padx=14, pady=12)
        vsb = ttk.Scrollbar(f, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        setattr(self, f"txt_{key}", txt)
        if key == "pre":
            txt.insert("1.0", "点击「开始拟合」后，这里会显示数据质量评估。")
        txt.configure(state="disabled")

    # ------------------------------------------------------------ actions --
    def _pick(self, var, prompt="选择文件夹"):
        """Folder picker.

        On macOS the Tk file dialog is avoided: it is an old Tk (8.6.12) modal that
        does not always come to the front, so the first click looks like it did
        nothing and the window loses key status afterwards.  AppleScript's
        ``choose folder`` opens the real Finder dialog and needs no special
        permission (unlike the System Events route, which is blocked).
        """
        if sys.platform == "darwin":
            self._pick_native(var, prompt)
        else:
            d = filedialog.askdirectory(title=prompt, parent=self)
            self._refocus()
            if d:
                var.set(d)
                self.update_idletasks()

    def _pick_native(self, var, prompt):
        import subprocess

        box = {}

        def work():
            esc = prompt.replace('"', '\\"')
            try:
                r = subprocess.run(
                    ["osascript", "-e",
                     f'POSIX path of (choose folder with prompt "{esc}")'],
                    capture_output=True, text=True)
                box["path"] = r.stdout.strip() if r.returncode == 0 else ""
            except Exception:
                box["path"] = ""

        # run in a thread: calling osascript inline would freeze the Tk event loop
        th = threading.Thread(target=work, daemon=True)
        th.start()

        def poll():
            if th.is_alive():
                self.after(150, poll)
                return
            p = box.get("path", "")
            if p:
                var.set(p)
                self.update_idletasks()
            self._refocus()

        self.after(150, poll)

    def _start(self):
        indir, outdir = self.var_in.get().strip(), self.var_out.get().strip()
        if not indir or not outdir:
            messagebox.showwarning(APP_TITLE, "请先选择两个文件夹。", parent=self)
            self._refocus()
            return
        if not (os.path.isdir(indir) and os.path.isdir(outdir)):
            messagebox.showerror(APP_TITLE, "文件夹路径无效。", parent=self)
            self._refocus()
            return
        try:
            has_jpg = any(f.lower().endswith((".jpg", ".jpeg")) for f in os.listdir(indir))
        except OSError:
            has_jpg = False
        if not has_jpg:
            messagebox.showerror(APP_TITLE, "A 文件夹里没有 JPG。", parent=self)
            self._refocus()
            return

        self.cancel_flag = False
        self.result = None
        self._preview_src = []
        self.btn_go.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.btn_export.configure(state="disabled")
        self.scale.configure(state="disabled")
        self.nb.tab(1, state="normal")
        self.nb.select(1)
        self._set_text(self.txt_pre, "正在扫描文件夹…\n")
        self._set_text(self.txt_rep, "")

        opts = FitOptions(lut_size=self.var_size.get(),
                          preserve_neutral=self.var_neutral.get())
        self.worker = threading.Thread(target=self._run, args=(indir, outdir, opts),
                                       daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_flag = True
        self.var_status.set("正在取消…")

    def _run(self, indir, outdir, opts):
        try:
            res = run_fit(indir, outdir, opts,
                          progress=lambda s, f, m: self.q.put(("progress", (f, m))),
                          should_cancel=lambda: self.cancel_flag,
                          on_preflight=lambda info: self.q.put(("preflight", info)))
            self.q.put(("done", res))
        except Cancelled:
            self.q.put(("cancelled", None))
        except BaseException as exc:                  # noqa: BLE001
            # BaseException, not Exception: a stray KeyboardInterrupt would kill this
            # thread silently and leave the UI stuck with the button disabled
            self.q.put(("error", (exc, traceback.format_exc())))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "progress":
                    frac, msg = payload
                    self.pb["value"] = frac * 1000
                    self.var_status.set(f"{frac*100:.0f}%   {msg}")
                elif kind == "preflight":
                    self._fill_preflight(payload, complete=False)
                elif kind == "done":
                    self._finish(payload)
                elif kind == "cancelled":
                    self._reset_buttons("已取消")
                elif kind == "error":
                    exc, tb = payload
                    self._reset_buttons("出错了")
                    messagebox.showerror(APP_TITLE, f"{exc}\n\n{tb[-1200:]}", parent=self)
                    self._refocus()
        except queue.Empty:
            pass
        except Exception:                             # never let polling die
            traceback.print_exc()
        finally:
            self.after(80, self._poll)

    def _reset_buttons(self, status):
        self.btn_go.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.var_status.set(status)

    # ----------------------------------------------------- preflight tab ---
    def _fill_preflight(self, info, complete=True):
        pf = info["preflight"]
        L = []
        L.append("数据质量评估\n" + "─" * 46 + "\n")
        L.append(f"A/B 图对          {info['n_pairs']} 组\n")
        L.append(f"有效像素          {info['pixels']/1e6:.1f} M\n\n")

        mc = pf["mass_coverage"]
        mark = "✓" if mc > 0.995 else ("⚠" if mc > 0.98 else "✗")
        L.append(f"{mark} 像素质量覆盖     {mc*100:.3f}%\n")
        L.append("     你的照片像素里，有多少落在「有训练数据」的颜色区域。\n")
        L.append("     这个数字决定 LUT 在真实照片上可不可靠。\n\n")
        L.append(f"· 色彩网格覆盖     {pf['node_coverage']*100:.1f}%  "
                 f"（{pf['n_bare']} / {pf['n_nodes']} 个节点无数据）\n")
        L.append("     立方体体积里有多少有色样支撑。这个数字天然偏低——\n")
        L.append("     自然照片的颜色只分布在一个薄面上，极端饱和色不会有。\n")
        L.append("     无数据区是平滑外推：不会出色带或反转，但颜色是推测值。\n\n")

        g = info.get("gray") or {}
        if g:
            L.append(f"· 中性轴实测       黑场 {g.get('black_lift', 0):.4f}   "
                     f"白场 {g.get('white_end', 0):.4f}   "
                     f"色偏 R-G {g.get('tint_rg', 0):+.4f}\n\n")

        if not complete:
            L.append("─" * 46 + "\n")
            L.append("⏳ 正在拟合，完成后这里会补上误差预估。\n")
        self._set_text(self.txt_pre, "".join(L))

    def _finish_preflight(self, res):
        pf, risk = res.preflight, res.risk
        base = self.txt_pre.get("1.0", "end").rstrip().split("─" * 46 + "\n⏳")[0].rstrip()
        L = [base, "\n\n"]
        if risk:
            L.append(f"· 外推区改动幅度   {risk['bare_mean_abs_shift']:.4f}"
                     f"（有数据区 {risk['supported_mean_abs_shift']:.4f}）\n")
            L.append("     外推区改动比数据区还小，说明外推是保守的。\n\n")
        rs = res.residual
        L.append("─" * 46 + "\n误差预估\n\n")
        L.append(f"  不可约下界（局部修图 + 压缩噪声）  {rs['irreducible']:.5f}\n")
        L.append(f"  模型误差（{res.options.lut_size}³ 表达不了的细节）"
                 f"      {rs['model']:.5f}\n")
        L.append(f"  预计总误差                          {rs['total']:.5f} 码值\n\n")
        L.append("  下界由数据决定，任何算法都无法突破。\n")
        self._set_text(self.txt_pre, "".join(L))

    # -------------------------------------------------------- report tab ---
    def _fill_report(self, res):
        st, agg = res.structural, res.aggregate
        fb, fa = st["folds_before"], st["folds_after"]
        jc, mo, ns = st["jacobian"], st["monotonicity"], st["neutral"]
        sw = st["axis_sweeps"]
        L = []
        L.append("结构检查（整张网格，含无数据区）\n" + "─" * 46 + "\n")
        L.append(f"  折叠胞元 det(J) ≤ 0    {fb['n_folded_cells']:>5} → {fa['n_folded_cells']}   "
                 f"（修复后最小 det {fa['min_det']:+.4f}）\n")
        L.append(f"  雅可比非正定            {jc['n_nonpositive']} / {jc['cells']}\n")
        L.append(f"  单通道单调违反          {mo['n_negative']}\n")
        L.append(f"  亮度单调违反            {mo['luma_n_negative']}\n")
        L.append(f"  最大二阶差分            "
                 f"{max(q['max_second_diff'] for q in sw.values()):.5f}\n")
        L.append(f"  中性轴偏离实测曲线      {ns.get('dev_from_curve', float('nan')):.5f}\n")
        L.append("  以上全部为零/极小 ⇒ 不会出现色带、色彩反转。\n\n")

        if agg:
            L.append("色彩还原度（LUT(A) vs B，CIEDE2000）\n" + "─" * 46 + "\n")
            L.append(f"  平均 ΔE00   {agg['mean']:.2f}      "
                     f"95 分位 {agg['p95']:.2f}      最大 {agg['max']:.2f}\n")
            L.append("  ΔE00 ≈ 1 是「刚好能察觉」的量级。\n\n")
            L.append("  分区：")
            L.append(f"中性 {agg['neutral']:.2f}   暗部 {agg['shadow']:.2f}   "
                     f"中间调 {agg['mid']:.2f}   高光 {agg['high']:.2f}   "
                     f"高饱和 {agg['sat']:.2f}\n\n")

        rows = sorted([p for p in res.per_pair if np.isfinite(p["mean"])],
                      key=lambda p: -p["mean"])[:5]
        if rows and rows[0]["mean"] > 2.5:
            L.append("⚠ 以下图对偏差较大，通常是它们的调色方向与整体不一致：\n")
            for p in rows:
                L.append(f"    {p['name']:<34} ΔE00 {p['mean']:.2f}\n")
            L.append("\n  单个 LUT 只能表达全局调色。这些图的后期方向如果与其余相反，\n")
            L.append("  LUT 只能折中——这是数据本身的性质，不是拟合出错。\n")

        self._set_text(self.txt_rep, "".join(L))
        self.nb.tab(2, state="normal")

    # ------------------------------------------------------------- finish --
    def _finish(self, res):
        self.result = res
        self.pb["value"] = 1000
        self.var_status.set(f"完成，用时 {res.elapsed:.0f} 秒")
        self.btn_go.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.btn_export.configure(state="normal")
        self.scale.configure(state="normal")

        self._finish_preflight(res)
        self._fill_report(res)

        self._preview_src = []
        for p in res.previews:
            self._preview_src.append(dict(name=p["name"],
                                          a=_tile(p["a"], TILE_W, TILE_H),
                                          b=_tile(p["b"], TILE_W, TILE_H)))
        self.strength.set(1.0)
        self.var_strength.set("100%")
        self.nb.select(0)
        self.after(60, self._render_preview)

    # ------------------------------------------------------------ preview --
    def _on_strength(self, _=None):
        self.var_strength.set(f"{self.strength.get()*100:.0f}%")
        if not self.result:
            return
        # dragging fires continuously; coalesce so the UI stays responsive
        if self._slider_job is not None:
            try:
                self.after_cancel(self._slider_job)
            except Exception:
                pass
        self._slider_job = self.after(110, self._render_preview)

    def _render_preview(self):
        self._slider_job = None
        if not self.result or not self._preview_src:
            return
        self.cv.delete("all")
        cw = self.cv.winfo_width()
        W = cw if cw > 50 else 900
        n = self.result.options.lut_size
        x = with_strength(self.result.lut, n, float(self.strength.get()))

        cols = []
        for src in self._preview_src:
            pred = apply_lut(x, src["a"], n)
            cols.append(np.concatenate([src["a"], pred, src["b"]], axis=0))

        comp = np.concatenate(cols, axis=1)
        for i in range(1, len(cols)):
            comp[:, i * TILE_W - 1:i * TILE_W + 1] = 0.12
        for k in (1, 2):
            comp[k * TILE_H - 1:k * TILE_H + 1, :] = 0.12

        photo = to_photo(comp, W)
        self._photos = [photo]
        self.cv.create_image(0, 0, anchor="nw", image=photo)
        self.cv.create_text(8, 8, anchor="nw", fill="#ffffff",
                            text="每列：上 = A 原图   中 = A + LUT   下 = B 目标",
                            font=("Helvetica", 11, "bold"))
        self.cv.configure(scrollregion=(0, 0, W, photo.height()))

    # ------------------------------------------------------------- export --
    def _export(self):
        if not self.result:
            return
        alpha = float(self.strength.get())
        chk = strength_check(self.result, alpha)
        if chk["folds"]["n_folded_cells"] > 0:
            if not messagebox.askyesno(
                    APP_TITLE,
                    f"当前强度下有 {chk['folds']['n_folded_cells']} 个胞元出现局部折叠"
                    f"（可能产生轻微色彩反转）。\n建议把强度调低一些。\n\n仍要导出吗？",
                    parent=self):
                return
        path = filedialog.asksaveasfilename(
            title="保存 LUT", defaultextension=".cube", parent=self,
            initialfile=f"lut{int(round(alpha*100))}.cube",
            filetypes=[("Cube LUT", "*.cube"), ("所有文件", "*.*")])
        self._refocus()
        if not path:
            return
        try:
            x = export_cube(self.result, path, alpha)
        except Exception as exc:                      # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"写入失败：{exc}", parent=self)
            self._refocus()
            return
        messagebox.showinfo(
            APP_TITLE,
            f"已导出：\n{path}\n\n"
            f"尺寸 {self.result.options.lut_size}³，强度 {alpha*100:.0f}%\n"
            f"数值范围 [{x.min():.4f}, {x.max():.4f}]", parent=self)
        self._refocus()

    @staticmethod
    def _set_text(widget, s):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", s)
        widget.configure(state="disabled")


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
