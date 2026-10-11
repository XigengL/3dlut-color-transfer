"""二分定位「点两次」：把 app.py 的各层逐个加上，看哪一层开始出问题。

用法（每次都从**终端**运行，结果同时打印到终端）：

    python3 tools/click_bisect.py plain     # 最朴素的 Tk
    python3 tools/click_bisect.py scroll    # + ScrollFrame
    python3 tools/click_bisect.py dialog    # + 文件夹对话框  ← 先跑这个
    python3 tools/click_bisect.py layout    # + Notebook + 底部固定区 + 滚轮绑定

每个模式里请**每个控件各点一次，看清楚再点**。终端会写明点击落到了哪个控件。
如果某个控件点了一次终端什么都没打印，那次点击就是被系统吞掉了 —— 这本身就是答案。
"""
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk

MODE = (sys.argv[1] if len(sys.argv) > 1 else "plain").lower()
BG = "#1e1f22"

root = tk.Tk()
root.title(f"click bisect  mode={MODE}")
root.geometry("680x560")
root.configure(bg=BG)

T0 = time.time()
NAMES = {}


def reg(widget, name):
    NAMES[str(widget)] = name


def make_widgets(parent, note):
    b1 = ttk.Button(parent, text="按钮 A", command=lambda: note(">>> command: 按钮 A"))
    b1.pack(pady=5)
    ent = ttk.Entry(parent)
    ent.pack(pady=5, fill="x", padx=40)
    b2 = ttk.Button(parent, text="按钮 B", command=lambda: note(">>> command: 按钮 B"))
    b2.pack(pady=5)
    reg(b1, "按钮A")
    reg(ent, "输入框")
    reg(b2, "按钮B")
    return b1, ent, b2


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfigure(self._win, width=e.width))


def build():
    ttk.Label(root, text=f"模式 = {MODE}    Tk {root.tk.call('info','patchlevel')}",
              font=("Helvetica", 13, "bold")).pack(pady=(10, 2))

    if MODE == "layout":
        # 和 app.py 一样：底部固定区先占位，中间是可滚动的 Notebook
        foot = ttk.Frame(root)
        foot.pack(side="bottom", fill="x", padx=14, pady=8)
        ttk.Label(foot, text="（这是模拟 app.py 的底部固定区）").pack(anchor="w")
        ttk.Progressbar(foot, maximum=1000, value=300).pack(fill="x")

        host = ScrollFrame(root)
        host.pack(side="top", fill="both", expand=True, padx=14, pady=6)
        parent = ttk.Frame(host.inner)
        parent.pack(fill="both", expand=True)

        nb = ttk.Notebook(parent, height=300)
        nb.pack(fill="both", expand=True)
        tab = ttk.Frame(nb)
        nb.add(tab, text="主页面")
        parent = tab

        # 和 app.py 一样的全局滚轮绑定
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.bind_all(seq, lambda e: None, add="+")
    elif MODE == "scroll":
        host = ScrollFrame(root)
        host.pack(fill="both", expand=True, padx=10)
        parent = ttk.Frame(host.inner)
        parent.pack(fill="both", expand=True, padx=6, pady=6)
    elif MODE == "dialog":
        parent = ttk.Frame(root)
        parent.pack(fill="both", expand=True, padx=10)
    else:
        parent = ttk.Frame(root)
        parent.pack(fill="both", expand=True, padx=10)

    log = tk.Text(parent, height=11, font=("Menlo", 11), bg="#17181a", fg="#e6e6e6")
    log.pack(side="bottom", fill="both", expand=True, pady=(8, 4))
    reg(log, "日志框")

    def note(s):
        line = f"{time.time() - T0:7.2f}s  {s}"
        try:
            log.insert("end", line + "\n")
            log.see("end")
            log.update_idletasks()
        except tk.TclError:
            pass
        print(line, flush=True)

    if MODE == "dialog":
        btn = ttk.Button(parent, text="① 点这个打开文件夹对话框")

        def do_pick():
            note("  >> 调用 askdirectory，对话框应该立刻出现在最前面")
            d = filedialog.askdirectory(title="选择文件夹", parent=root)
            note(f"  << 对话框返回: {d!r}")
            root.lift()
            root.focus_force()
            note("  << 已尝试重新抢焦点")

        btn.configure(command=do_pick)
        btn.pack(pady=5)
        reg(btn, "打开对话框按钮")
        note("先点「① 打开文件夹对话框」一次，关掉后再点 按钮A / 输入框 / 按钮B")

    make_widgets(parent, note)
    return note


note = build()


def on_press(e):
    note(f"press    -> {NAMES.get(str(e.widget), e.widget.winfo_class())}")


def on_release(e):
    note(f"release  -> {NAMES.get(str(e.widget), e.widget.winfo_class())}")


root.bind_all("<ButtonPress-1>", on_press, add="+")
root.bind_all("<ButtonRelease-1>", on_release, add="+")

note(f"就绪。模式 {MODE}")
root.mainloop()
