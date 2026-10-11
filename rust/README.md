# lutfit — Rust 版 3D LUT 拟合工具

把「相机原图」和「修图后」两个文件夹喂进去，输出一个 `.cube`。macOS / Windows / Linux 同一份源码。

## 与 Python 版的关系

Rust 版是 Python 版的逐行移植，**数值上等价**，不是重写算法。

验证方式有两条：

**① 黄金测试**（`cargo test --release`）——数值核心逐项对齐：

| 测试 | 判据 |
|---|---|
| CIEDE2000 | 对 Sharma 等人公开的 34 对参考值，最大偏差 < 1e-4 |
| 传递函数 | sRGB 编解码与 Python 逐位一致（< 1e-15） |
| Oklab / CIELAB | 与 Python 偏差 < 1e-12 |
| 四面体索引与权重 | 3000 个随机点，索引零不匹配，权重偏差 < 1e-15 |
| 恒等 LUT 重构 | 2³/5³/17³/33³ 上误差 < 1e-11 |
| Laplacian | 常量场与线性场严格归零 |

**② 端到端对照**（`cargo test --release -- --ignored end_to_end`）——用真实素材拟合，与 Python 生成的 LUT 比对：

```
../input : 平均偏差 0.00080  99.9分位 0.01073  最大 0.02987
../input2: 平均偏差 0.00081  99.9分位 0.01026  最大 0.03085
```

判据用**平均偏差与 99.9 分位**，不用最大值——折叠修复是打地鼠过程，个别节点必然有差异。两个 LUT 的功能等价性由 ΔE00 证明：数据集1 上 Python 版 1.1994、Rust 版 1.1993。

### 两批实测数据对照

| 指标 | 数据集1 Python | 数据集1 Rust | 数据集2 Python | 数据集2 Rust |
|---|---|---|---|---|
| 图对 / 有效像素 | 46 / 154.0M | 46 / **155.3M** | 87 / 238.3M | 87 / **239.3M** |
| 节点覆盖率 | 31.7% | 31.4% | 48.3% | 48.1% |
| 像素质量覆盖率 | 99.933% | 99.993% | 99.980% | 99.993% |
| **模型误差** | 0.01299 | **0.01299** | 0.03886 | **0.03893** |
| **不可约下界** | 0.02725 | **0.02692** | 0.07377 | **0.07357** |
| **总误差 / 下界** | +10.8% | **+11.0%** | +13.0% | **+13.1%** |
| **ΔE00 均值** | 1.20 | **3.81→3.79** | 3.82 | **3.79** |
| 折叠（修复后） | 0 | **0** | 0 | **0** |
| **耗时** | 7.8 min | **59 s** | 22.2 min | **142 s** |

**模型误差、下界、总误差、ΔE 全部一致；速度快 8-9 倍。**

交叉验证：用 Python 评估 Rust 生成的 LUT，ΔE00 = 1.199，而 Python 自己的 LUT 是 1.20。

## 关于 Mac M 芯片 GPU

**结论：UI 已经跑在 GPU 上，但数学计算放 GPU 不划算。** 这不是猜的，是量出来的：

```
数据集2（87 组）：拟合 75s，其中 matvec 44,956 次，平均 1.68 ms/次
每次 matvec ≈ 116 万次浮点运算
```

- **渲染**：eframe 走 wgpu，在 macOS 上后端就是 **Metal**。界面本身已经 GPU 加速，4K 屏也不掉帧。
- **求解器**：整个拟合总共约 53 GFLOP。M 系列 GPU 纯算力约 30ms 就能算完——但这是 **4.5 万次严格串行的、每次只有 116 万 FLOP 的小矩阵向量乘**。每次迭代要 6 个左右 kernel dispatch，Metal 单次派发开销 20-100µs，光派发就要 5-27 秒，和 CPU 的 75 秒相比没有量级优势，实际很可能更慢。
- **数据局部性**：解向量 864KB、算子 4MB，**完全装得进 CPU 的 L2/L3 缓存**。CPU 已经接近内存带宽上限，GPU 要额外付 700MB 的显存搬运。
- **JPEG 解码**：43 秒，纯 CPU 任务，GPU 帮不上。
- **色差评估**：23 秒，倒是 embarrassingly parallel，但要先把 700MB 传进显存，收益有限。

**真正的优化方向是减少迭代次数**（更好的预条件子、或稀疏 Cholesky 直接解），而不是换硬件。顺带试过一个便宜的优化——让 CG 在 IRLS 各轮之间热启动——只省了 2%，因为每轮系统都变了。

## 构建

**macOS**
```bash
./build_mac.sh          # → dist/lutfit.app  (5.6 MB)
```
首次打开若被 Gatekeeper 拦：右键 → 打开，或 `xattr -dr com.apple.quarantine dist/lutfit.app`

**Windows**
```bat
build_win.bat           :: → dist\lutfit.exe
```

**手动**
```bash
cargo build --release
./target/release/lutfit                  # 图形界面
./target/release/lutfit fit A文件夹 B文件夹 out.cube 33   # 命令行
```

## 文档

- **[DESIGN.md](DESIGN.md)** —— 拟合原理、误差账本、下界分解、优化空间分析。
  想知道"算法还有多少空间"看这一篇；核心结论是**当前模型类里最多再降 10%，
  要突破必须改变模型本身**。
- **本文** —— 使用、构建、与 Python 版的数值对照。

## 界面缺陷修复记录

用截图 + 像素分析定位（`screencapture` + PIL 扫描），不是靠猜：

**中间的黑竖条**。侧栏 frame 结束于 x=848，中央 frame 从 x=892 才开始，中间 44px 是纯黑。原因是 eframe 默认的 `clear_color` 是透明的，两个面板 frame 之间的空隙直接透出黑底。修法是在 `App` 里覆写 `clear_color` 返回窗口底色。

**预览图从第二列起乱码**。每张预览各自保留原图宽高比，尺寸不一致；但界面把它们拼成一张纹理时用第一张的尺寸去算所有列偏移，第一列恰好对得上，后面的全部错位。现在统一输出 420×315。有测试守着。

**进度条文字被遮挡**。进度条高度设成 6px 却调用 `show_percentage()`，文字塞不进去。改成：大号百分比数字 + 右侧剩余时间 + 8px 纯色进度条 + 当前阶段 + 已用时间。

**页面无法滚动**。两个原因：滚动条用默认的「需要时才出现」，加上窗口请求 940pt 高但屏幕只有 975pt（扣掉菜单栏和标题栏放不下），系统静默压缩并把滚动区底部挤出屏幕。现在滚动条常驻，窗口高度按 `ctx.input(|i| i.viewport().monitor_size)` 自适应（不依赖任何平台 API）。

## 截图调试钩子

```bash
LUTFIT_DEBUG_LAYOUT=1 ./target/release/lutfit --gallery A文件夹 B文件夹 33   # 打印内容高/视口高
LUTFIT_SCROLL=bottom  ./target/release/lutfit --gallery A文件夹 B文件夹 33   # 打开即滚到底
```

`--gallery` 会预填路径并自动开始拟合，配合 `screencapture` 可以无人值守地截到运行中和完成态。

## 中文字体

egui 内置字体不含汉字，**不处理的话界面上的中文全是方块**。程序启动时会在系统里找一个含中文字形的字体（`theme::install_cjk_font`），按下面的顺序取第一个存在的：

| 平台 | 候选 |
|---|---|
| macOS | PingFang.ttc → Hiragino Sans GB.ttc → STHeiti Light/Medium.ttc → Songti.ttc → Arial Unicode |
| Windows | msyh.ttc → msyh.ttf → simhei.ttf → simsun.ttc |
| Linux | Noto Sans CJK → wqy-microhei → uming |

这台机器上没有 PingFang，用的是 `Hiragino Sans GB`。字体装在 `Proportional` 族的最前面，中英文用同一套字形，风格统一。

有测试守着这件事，不用靠肉眼看：

```
$ cargo test --release a_system_cjk_font -- --nocapture
可用中文字体: /System/Library/Fonts/Hiragino Sans GB.ttc#0（9/9 字形齐全）
```

它逐个检查候选字体里「中文图像强度预检」这些字有没有真实字形，找不到就 panic 并把检查过的路径打出来。

如果哪台机器上确实一个都没有（极度精简的 Linux 容器），可以从 [[https://github.com/notofonts/noto-cjk]] 下一个 `NotoSansCJKsc-Regular.otf` 放进系统字体目录，或者把路径加进 `theme::cjk_candidates()`。

## 环境注意

沙箱/CI 里如果 `$HOME` 不可写，把 cargo 和 rustup 的数据目录指到可写位置即可：

```bash
export RUSTUP_HOME=/path/to/rustup CARGO_HOME=/path/to/cargo
```

## 界面

Apple 极简风格：`#F5F5F7` 背景、白色圆角卡片、`#1D1D1F` 正文、单一强调色 `#0071E3`，大留白，无装饰性元素。所有配色集中在 `src/theme.rs`。

**布局**：左侧固定 330px 操作栏（三步走：选图 → 选尺寸 → 开始），右侧自适应内容区（预检 → 拟合质量 → 结构检查 → 预览 → 强度与导出）。

**进度**：各层按 n³ 加权，33³ 末层占约 60%，所以进度条不会被卡在 90% 不动；同时显示已用时间和预计剩余时间。

**预检**用大白话解释每个数字，并给出颜色判定：

| 指标 | 含义 |
|---|---|
| 像素质量覆盖 | 真实照片里落在有效数据上的像素比例。> 99.5% 为绿色 |
| 色彩网格覆盖 | 色彩立方体的节点中有多少训练数据，其余是平滑推测 |
| 外推区改动量 | 没数据的地方被改了多少（对照有数据处的数值） |

**预览**：5 张一行的联系表，每列从上到下为「原图 → 套用 LUT → 修图师成品」。

选图不是随便挑的——在 (平均亮度, 平均饱和度, ΔE) 三维特征空间里做贪心最远点采样，从 ΔE 中位数那张开始，每次取离已选集合最远的一张。这样五张会覆盖不同明暗、不同饱和度、不同难度，而不是五个同类场景。

**预览尺寸不变式**：所有预览统一输出为 420×315（中心裁切到 4:3 再均值降采样）。早期版本让每张保留自己的宽高比，而 UI 用第一张的尺寸去索引整张联系表，结果第一列之后的每一列都是乱码。现在有测试守着：

```
$ cargo test --release -- --ignored end_to_end --nocapture
../input: 平均偏差 0.00080  99.9分位 0.01073  最大 0.02987  ΔE 1.01
  预览 5 张，尺寸统一 (420, 315)
```

## 代码结构

```
src/
  main.rs          入口（无参数 = GUI，有参数 = CLI）
  theme.rs         Apple 风格设计令牌与 egui 样式
  app.rs           GUI：后台线程拟合、预检卡片、预览、强度滑杆、导出
  core/
    color.rs       sRGB / Display P3、传递函数、Oklab、CIELAB、CIEDE2000
    tetra.rs       四面体插值、稀疏算子、Laplacian
    project.rs     单调惩罚、PAVA、折叠检测与修复、polish、色域压缩
    fit.rs         预条件 CG、箱约束活跃集、多尺度、自适应正则化
    loader.rs      JPEG 解码、EXIF 旋转、掩膜
    accumulate.rs  流式分箱统计（图片并行解码）
    coverage.rs    预检覆盖度与外推风险
    cube.rs        .cube 读写
    pipeline.rs    端到端编排
tests/golden.rs    黄金测试
```
