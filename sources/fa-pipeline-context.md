# FA2 / FA3 / FA4 pipeline：同工作量同宽的最终 context

## 1. 当前图和用户确认的范围

用户要理解 pipeline 哪里被阻塞、后续设计如何缓解阻塞。用户明确要求：**按工作量表示长度，不考虑硬件性能带来的变化，不需要最终加速倍数。**

PNG：https://pxl.cl/gq7Xz

本地文件：
- `/tmp/fa4_learn/fa2_fa3_fa4_pipeline_equal_op_work.png`
- `/tmp/fa4_learn/fa2_fa3_fa4_pipeline_equal_op_work.svg`
- `/tmp/fa4_learn/gen_fa_pipeline_blockers.py`
- `/tmp/fa4_learn/render_svg.py`

PNG 3040×4230，SVG 1520×2115。本 paste 下方包含完整 SVG、生成器和 renderer，换 session / 主机也可恢复。

## 2. 必须保留的绘图假设

- 三图处理相同的**逻辑** A/B 两个 Q 行分片和连续两轮 KV，均含4份 QK、softmax、O 修正、PV 工作。它们是逻辑 tile 运算，不是4条硬件矩阵指令。
- A/B 是不同 query 行；0/1 是示例连续轮次，不是实际源码倒序扫描的 j 下标，也不是缓冲 slot 编号。
- 所画两轮在稳态中，之前轮次已有 O 累积。三图首轮 Q/K/V 均已就绪；启动、收尾简化。
- 所有版本中，同类、同工作量的操作用相同宽度，包括 FA4 的 QK/PV，以及下一轮 K/V 搬运。
- 不同类型操作间的相对长度是**示意权重**。生成器采用 QK/PV=2、softmax关键段=3、O修正=1、单次K/V搬运=2的绘图单位；它们不是实际 FLOPs、指令条数、周期或测量值。
- 因此横向长度用于说明这套假设下的重叠与依赖，**不能相除得出真实加速倍数**，也不代表所有 kernel 配置都有相同调度或瓶颈。
- 不引入 Blackwell Tensor Core 吞吐提升，不缩短 FA4 矩阵框；也不计 FMA exp、减少重缩放等算法减少工作量的收益。
- TMA、TMEM、异步矩阵指令仍保留为“允许这种角色分工的硬件能力”；这里只排除了硬件吞吐差异。
- 图是机制示意，不是精确源码 timeline / profiler trace。尤其 FA2 的 CTA 分阶段推进是抽象排布，不能推断硬件禁止 warp 间或 CTA 间重叠，也不能将示意中的空档称为实际测得的 stall。

## 3. 每张图表达什么

### FA2

同一批 warps 做搬运、QK、softmax、O重缩放、PV。将它按 CTA 的计算阶段展开时，非矩阵阶段缺少可发出的本 CTA 矩阵工作。PV 需要 softmax 产出 P，并保持 O 的在线累积状态正确。

cp.async 已能重叠部分搬运；其他 warp / CTA 也可以填补部分空档。红色不是“整个 GPU 必然空闲”。

### FA3

WG1/WG2 按 Q 行分工，各自做 QK → softmax/O修正 → PV。一组 softmax 时，另一组可以做矩阵工作；producer w0 用 TMA 预取。

FA3 的 WG 是参与 WGMMA 的硬件 warpgroup，4 warps/128线程。每组自己做 softmax 和 PV，不把这些操作交给单独的 softmax 角色。

FA3 既有组间交错，也支持组内 overlap；图主要展开组间部分。更多在途 S/P/O 占用 consumer 寄存器，限制流水线深度。

### FA4

w12 发 QK/PV；softmax0/softmax1 做 A/B 对应行的 softmax；correction 做 O重缩放；w14 发 TMA。矩阵中间结果通过 TMEM 交接，使 O修正可与 exp 关键段重叠，而不都放在同一 consumer WG 的工作路径中。

具体原 tiling 配置中的 roles：w0–3 softmax0，w4–7 softmax1，w8–11 correction，w12 MMA，w13 epilogue，w14 load，w15 empty；16个分配 warp 中15个工作。

“单 warp 发矩阵指令”不表示一个 warp 计算所有矩阵元素；Tensor Core 硬件异步执行。softmax warp 是发普通运算指令的线程组，不是 Tensor Core。

## 4. 不会消失的 blocker

- KV 数据就绪 → 才能执行对应 QK/PV。
- S 结果就绪 → softmax 才能处理；P与O状态就绪 → PV 才能开始。
- 多个 softmax warp 共用所在 SM 的指数资源，增加 warp 不等于增加指数单元吞吐。黄色块主要表示指数关键段；准备等阶段可重叠。
- 获得新 row max / 重缩放系数后，独立 correction 可与 exp 重叠。O修正没有消失，只是换了执行角色和时机。
- 缓冲数据消费完并确认释放 → slot 才能覆盖。FA4 三个 K-or-V slot 可以放当前 K、当前 V、下一 K；当前 K消费后，原槽装下一 V。
- TMA 行展示异步传输活动；w0/w14仅发请求，不表示它们全程忙于搬运或阻塞等待。
- 寄存器、SMEM/TMEM容量、带宽和资源竞争仍会限制可重叠多少。当前图不模拟所有这些约束，不用于判断实际最优 schedule。

## 5. 为什么前一版 FA3 比 FA2 画得长

此前 FA2 聚合成一份流程，FA3 展开 A/B 两份，横轴宽度又未统一，导致不能比较。已修正为同一逻辑工作量；不要恢复旧版 FA2只画一份、FA3画两份的排版。

最新版本又去掉了 FA4矩阵块更短的硬件性能假设。用户最后确认的是本 paste 的 equal_op_work 版本。

## 6. 绘图与复现约定

- 行表示执行角色或当前 CTA 对硬件的矩阵工作供给，列表示示意推进；同一操作在角色行与硬件行的重复显示是投影，不是重复计算。
- 红色斜纹直接放在依赖未满足/争用的区间；就绪箭头接到下一操作前，不把等待扩展成整个轮次。
- 操作框中保留数据归属 A/B、轮次0/1；右侧仅解释 blocker 和改善。
- PNG需保持 SVG 原始尺寸的2倍输出，保留可编辑 SVG；不要在改图时丢掉文字或让标签与框重叠。
- 直接保存下方 SVG 为 `fa2_fa3_fa4_pipeline_equal_op_work.svg`，renderer 保存为 `render_svg.py`，运行：

```bash
/usr/bin/python3 render_svg.py fa2_fa3_fa4_pipeline_equal_op_work.svg fa2_fa3_fa4_pipeline_equal_op_work.png
```

renderer 使用系统 librsvg / Cairo，经 Python ctypes 调用，不需要 PIL、cairosvg 或安装 pip 包。需有 `librsvg-2.so.2`、`libcairo.so.2`、`libgobject-2.0.so.0` 和支持中文的字体。

若需修改生成器，调整 `ROOT` 输出目录；它会同时生成 canonical、same_work 和 equal_op_work SVG。当前绘图 generator 中 FA3及之后元素有统一的纵向偏移处理，改布局时注意这一点。

## 7. 机制来源

- FA3：https://pytorch.org/blog/flashattention-3/ ，组间ping-pong、组内overlap和寄存器代价。
- FA4：https://arxiv.org/html/2603.05451v1 ，§2.2 TMEM/MMA，§3.1.2角色交接与correction解耦，§3.1.3–3.1.4指数和重缩放优化。
- 源码路径：FA2 `fbcode/ai_codesign/gen_ai/flash_attention_v2/csrc/flash_attn/src/`；FA3 `fbcode/fa3/hopper/`；FA4 `fbcode/ads_mkl/ops/cute_dsl/fa4/src/flash_fwd_sm100.py`。

此 context 不包含速度数字；没有运行本地 GPU benchmark。


## 8. 当前文件指纹

- `fa2_fa3_fa4_pipeline_equal_op_work.svg`：SHA256 `8a0316340fcdf79f142fd102494d016532b06334a590a1be6e74e87085ef574f`
- `fa2_fa3_fa4_pipeline_equal_op_work.png`：SHA256 `d19f4ce5d4f8eca6c0d721b937418729335e7c318a433db98a2eb758800a9b55`


## 完整 SVG

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="1520" height="2115" viewBox="0 0 1520 2115" font-family="Noto Sans CJK SC,Arial,sans-serif" role="img">
<title>FA2、FA3、FA4：pipeline 如何缓解阻塞，哪些等待仍然存在</title><desc>按逻辑时间展示计算、softmax、O修正和异步搬运。红色斜纹标出等待P、等待指数单元或存储槽的依赖。图是机制示意，不是实测时间线。</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b" /></marker><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#fef2f2" /><path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#fecaca" stroke-width="1" /></pattern></defs>
<rect x="0" y="0" width="1520" height="2115" rx="0" fill="#ffffff" stroke="#ffffff" />
<text x="30" y="42" font-size="29" fill="#0f172a" font-weight="700" text-anchor="start">FA2 → FA3 → FA4：pipeline 改善了哪里的 blocker？</text>
<text x="30" y="72" font-size="16" fill="#334155" font-weight="400" text-anchor="start">三图均处理同一组逻辑 A/B 分片 × 两轮 KV：每图均包含4份 QK / softmax / O 修正 / PV 工作。</text>
<text x="30" y="99" font-size="15" fill="#334155" font-weight="400" text-anchor="start">同类操作、相同工作量均同宽；不计硬件吞吐差异或软件减少运算量的收益。不同操作间的相对宽度仅作示意。</text>
<rect x="30" y="113" width="18" height="18" rx="3" fill="#dbeafe" stroke="#2563eb" />
<text x="55" y="128" font-size="14" fill="#334155" font-weight="400" text-anchor="start">QK / PV · Tensor Core</text>
<rect x="350" y="113" width="18" height="18" rx="3" fill="#fef3c7" stroke="#b45309" />
<text x="375" y="128" font-size="14" fill="#334155" font-weight="400" text-anchor="start">softmax · 指数 / ALU</text>
<rect x="660" y="113" width="18" height="18" rx="3" fill="#ffedd5" stroke="#ea580c" />
<text x="685" y="128" font-size="14" fill="#334155" font-weight="400" text-anchor="start">O 修正</text>
<rect x="835" y="113" width="18" height="18" rx="3" fill="#dcfce7" stroke="#16a34a" />
<text x="860" y="128" font-size="14" fill="#334155" font-weight="400" text-anchor="start">异步搬运</text>
<rect x="1070" y="113" width="18" height="18" rx="3" fill="url(#hatch)" stroke="#dc2626" />
<text x="1095" y="128" font-size="14" fill="#dc2626" font-weight="400" text-anchor="start">等待 / 资源竞争</text>
<rect x="30" y="155" width="1460" height="477" rx="7" fill="#ffffff" stroke="#cbd5e1" />
<text x="50" y="188" font-size="23" fill="#0f172a" font-weight="700" text-anchor="start">FA2：同一批 warps 做搬运和整套计算</text>
<text x="50" y="216" font-size="15" fill="#334155" font-weight="400" text-anchor="start">按 CTA 的计算阶段展开 A/B 两份工作；三图首轮 Q/K/V 均已就绪，启动和收尾省略。</text>
<line x1="235" y1="238" x2="1135" y2="238" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="1135" y="232" font-size="12" fill="#64748b" font-weight="400" text-anchor="end">执行推进 →</text>
<text x="50" y="284" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">warps · Q 分片 A</text>
<rect x="235" y="260" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<text x="50" y="364" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">warps · Q 分片 B</text>
<rect x="235" y="340" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="260" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="263.125" y="283.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A0</text>
<rect x="349.5" y="260" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="389.6875" y="275" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="389.6875" y="289" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A0</text>
<rect x="518.25" y="260" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="530.3125" y="275" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="530.3125" y="289" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">A0</text>
<rect x="574.5" y="260" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="600.625" y="283.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A0</text>
<rect x="293.25" y="340" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="319.375" y="363.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B0</text>
<rect x="433.875" y="340" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="474.0625" y="355" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="474.0625" y="369" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B0</text>
<rect x="546.375" y="340" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="558.4375" y="355" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="558.4375" y="369" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">B0</text>
<rect x="630.75" y="340" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="656.875" y="363.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B0</text>
<rect x="687.0" y="260" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="713.125" y="283.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A1</text>
<rect x="799.5" y="260" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="839.6875" y="275" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="839.6875" y="289" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A1</text>
<rect x="968.25" y="260" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="980.3125" y="275" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="980.3125" y="289" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">A1</text>
<rect x="1024.5" y="260" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="1050.625" y="283.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A1</text>
<rect x="743.25" y="340" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="769.375" y="363.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B1</text>
<rect x="883.875" y="340" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="924.0625" y="355" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="924.0625" y="369" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B1</text>
<rect x="996.375" y="340" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="1008.4375" y="355" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="1008.4375" y="369" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">B1</text>
<rect x="1080.75" y="340" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="1106.875" y="363.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B1</text>
<text x="50" y="444" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">本 CTA 的矩阵工作</text>
<rect x="235" y="420" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="263.125" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A0</text>
<rect x="574.5" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="600.625" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A0</text>
<rect x="293.25" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="319.375" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B0</text>
<rect x="630.75" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="656.875" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B0</text>
<rect x="687.0" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="713.125" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A1</text>
<rect x="1024.5" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="1050.625" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A1</text>
<rect x="743.25" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="769.375" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B1</text>
<rect x="1080.75" y="420" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="1106.875" y="443.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B1</text>
<rect x="349.5" y="420" width="221.0" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="460.0" y="443.5" font-size="12" fill="#dc2626" font-weight="700" text-anchor="middle">softmax / O 修正阶段：等 P₀</text>
<rect x="799.5" y="420" width="221.0" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="910.0" y="443.5" font-size="12" fill="#dc2626" font-weight="700" text-anchor="middle">softmax / O 修正阶段：等 P₁</text>
<text x="50" y="524" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">异步搬运</text>
<rect x="235" y="500" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="349.5" y="500" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="375.625" y="523.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 K₁</text>
<rect x="687.0" y="500" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="713.125" y="523.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 V₁</text>
<rect x="1160" y="256" width="310" height="229" rx="7" fill="#f8fafc" stroke="#cbd5e1" />
<text x="1176" y="285" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">blocker：下一次 PV 缺输入</text>
<text x="1176" y="317" font-size="15" fill="#334155" font-weight="400" text-anchor="start">QK 算完不等于 PV 可以开始。</text>
<text x="1176" y="344" font-size="15" fill="#334155" font-weight="400" text-anchor="start">必须先得到 softmax 的 P，</text>
<text x="1176" y="371" font-size="15" fill="#334155" font-weight="400" text-anchor="start">并完成对应 O 的重缩放。</text>
<text x="1176" y="398" font-size="15" fill="#334155" font-weight="400" text-anchor="start">同批 warps 此时做非矩阵运算。</text>
<text x="1176" y="425" font-size="15" fill="#334155" font-weight="400" text-anchor="start">红色是本 CTA 的供给空档；</text>
<text x="1176" y="452" font-size="15" fill="#334155" font-weight="400" text-anchor="start">其他 CTA 可填补，不等于全 GPU 闲置。</text>
<text x="50" y="588" font-size="15" fill="#334155" font-weight="400" text-anchor="start">cp.async 已能重叠部分搬运；改进方向是让一份 Q 分片的非矩阵阶段，与另一份的矩阵工作交错。</text>
<text x="50" y="616" font-size="15" fill="#334155" font-weight="400" text-anchor="start">红色展示阶段式推进的供给空档；不表示 FA2 硬件禁止不同 warp 或不同 CTA 重叠计算。</text>
<rect x="30" y="657.0" width="1460" height="490" rx="7" fill="#ffffff" stroke="#cbd5e1" />
<text x="50" y="690.0" font-size="23" fill="#0f172a" font-weight="700" text-anchor="start">FA3：两个 consumer WG 交错，用另一组的矩阵工作填空档</text>
<text x="50" y="718.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">每组仍负责 QK、softmax、O 修正、PV；此图主要展开组间交错，FA3 也支持组内重叠。</text>
<line x1="235" y1="740.0" x2="1135" y2="740.0" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="1135" y="734.0" font-size="12" fill="#64748b" font-weight="400" text-anchor="end">执行推进 →</text>
<text x="50" y="786.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">WG1 · Q 分片 A</text>
<rect x="235" y="762.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="762.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="263.125" y="785.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₀</text>
<rect x="293.25" y="762.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="333.4375" y="777.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="333.4375" y="791.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A₀</text>
<rect x="377.625" y="762.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="389.6875" y="785.5" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">修正</text>
<rect x="405.75" y="762.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="431.875" y="785.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₀</text>
<rect x="462.0" y="762.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="488.125" y="785.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₁</text>
<rect x="518.25" y="762.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="558.4375" y="777.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="558.4375" y="791.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A₁</text>
<rect x="602.625" y="762.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="614.6875" y="785.5" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">修正</text>
<rect x="630.75" y="762.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="656.875" y="785.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₁</text>
<text x="50" y="866.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">WG2 · Q 分片 B</text>
<rect x="235" y="842.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="293.25" y="842.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="319.375" y="865.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₀</text>
<rect x="405.75" y="842.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="445.9375" y="857.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="445.9375" y="871.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B₀</text>
<rect x="490.125" y="842.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="502.1875" y="865.5" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">修正</text>
<rect x="518.25" y="842.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="544.375" y="865.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₀</text>
<rect x="574.5" y="842.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="600.625" y="865.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₁</text>
<rect x="630.75" y="842.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="670.9375" y="857.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="670.9375" y="871.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B₁</text>
<rect x="715.125" y="842.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="727.1875" y="865.5" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">修正</text>
<rect x="743.25" y="842.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="769.375" y="865.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₁</text>
<rect x="349.5" y="842.0" width="52.25" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="375.625" y="865.5" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">等调度 / 依赖</text>
<text x="50" y="946.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">本 CTA 的矩阵工作</text>
<rect x="235" y="922.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="263.125" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₀</text>
<rect x="293.25" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="319.375" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₀</text>
<rect x="405.75" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="431.875" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₀</text>
<rect x="462.0" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="488.125" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₁</text>
<rect x="518.25" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="544.375" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₀</text>
<rect x="574.5" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="600.625" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₁</text>
<rect x="630.75" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="656.875" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₁</text>
<rect x="743.25" y="922.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="769.375" y="945.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₁</text>
<rect x="349.5" y="922.0" width="52.25" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="375.625" y="945.5" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">等 P / 修正</text>
<rect x="687.0" y="922.0" width="52.25" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="713.125" y="945.5" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">等 P / 修正</text>
<line x1="319.375" y1="803.0" x2="319.375" y2="919.0" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" />
<text x="329.375" y="899.0" font-size="12" fill="#0d9488" font-weight="400" text-anchor="start">WG1 softmax ∥ WG2 QK</text>
<text x="50" y="1026.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">TMA · w0 发请求</text>
<rect x="235" y="1002.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="1002.0" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="263.125" y="1025.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 K₁</text>
<rect x="293.25" y="1002.0" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="319.375" y="1025.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 V₁</text>
<rect x="1160" y="757.0" width="310" height="256" rx="7" fill="#f8fafc" stroke="#cbd5e1" />
<text x="1176" y="786.0" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">改善：用交错调度隐藏等待</text>
<text x="1176" y="818.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">WG1 做 softmax 时，</text>
<text x="1176" y="845.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">WG2 可以发矩阵工作。</text>
<text x="1176" y="872.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">TMA 搬运也独立推进。</text>
<text x="1176" y="899.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">仍要等各组自己的 P / O 就绪。</text>
<text x="1176" y="926.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">组内进一步重叠，需要同时</text>
<text x="1176" y="953.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">持有更多 S / P / O 寄存器。</text>
<text x="1176" y="980.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">寄存器压力限制流水线深度。</text>
<text x="50" y="1095.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">WGMMA 是异步的；“每组做完整流程”不代表硬件要求该组所有操作严格串行。</text>
<text x="50" y="1124.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">更多重叠仍需足够寄存器保存尚未消费的结果；下一数据不能覆盖还在使用的缓冲。</text>
<rect x="30" y="1172.0" width="1460" height="570" rx="7" fill="#ffffff" stroke="#cbd5e1" />
<text x="50" y="1205.0" font-size="23" fill="#0f172a" font-weight="700" text-anchor="start">FA4：TMEM 交接结果，矩阵乘、softmax、O 修正分给不同角色</text>
<text x="50" y="1233.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">矩阵、softmax、O 修正工作量与前两图一致；只展开角色解耦带来的调度变化。</text>
<line x1="235" y1="1255.0" x2="1135" y2="1255.0" stroke="#64748b" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="1135" y="1249.0" font-size="12" fill="#64748b" font-weight="400" text-anchor="end">执行推进 →</text>
<text x="50" y="1301.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">w12 / Tensor Core</text>
<rect x="235" y="1277.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="263.125" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₀</text>
<rect x="293.25" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="319.375" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₀</text>
<rect x="377.625" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="403.75" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₀</text>
<rect x="433.875" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="460.0" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK A₁</text>
<rect x="490.125" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="516.25" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₀</text>
<rect x="546.375" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="572.5" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">QK B₁</text>
<rect x="602.625" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="628.75" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV A₁</text>
<rect x="687.0" y="1277.0" width="52.25" height="39" rx="5" fill="#dbeafe" stroke="#2563eb" />
<text x="713.125" y="1300.5" font-size="12" fill="#2563eb" font-weight="700" text-anchor="middle">PV B₁</text>
<rect x="349.5" y="1277.0" width="24.125" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="361.5625" y="1300.5" font-size="10" fill="#dc2626" font-weight="700" text-anchor="middle">P</text>
<rect x="658.875" y="1277.0" width="24.125" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="670.9375" y="1300.5" font-size="10" fill="#dc2626" font-weight="700" text-anchor="middle">P</text>
<text x="50" y="1381.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">softmax0 · A</text>
<rect x="235" y="1357.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="293.25" y="1357.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="333.4375" y="1372.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="333.4375" y="1386.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A₀ → PA₀</text>
<rect x="490.125" y="1357.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="530.3125" y="1372.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="530.3125" y="1386.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">A₁ → PA₁</text>
<text x="50" y="1461.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">softmax1 · B</text>
<rect x="235" y="1437.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="349.5" y="1437.0" width="24.125" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="361.5625" y="1460.5" font-size="10" fill="#dc2626" font-weight="700" text-anchor="middle">EXP</text>
<rect x="377.625" y="1437.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="417.8125" y="1452.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="417.8125" y="1466.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B₀ → PB₀</text>
<rect x="602.625" y="1437.0" width="80.375" height="39" rx="5" fill="#fef3c7" stroke="#b45309" />
<text x="642.8125" y="1452.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">softmax</text>
<text x="642.8125" y="1466.0" font-size="10" fill="#b45309" font-weight="700" text-anchor="middle">B₁ → PB₁</text>
<line x1="375.625" y1="1354.0" x2="375.625" y2="1319.0" stroke="#dc2626" stroke-width="1.5" marker-end="url(#arrow)" />
<text x="383.625" y="1336.0" font-size="12" fill="#dc2626" font-weight="400" text-anchor="start">P A₀ 就绪 → PV A₀</text>
<text x="50" y="1541.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">correction · O 修正</text>
<rect x="235" y="1517.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="321.375" y="1517.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="333.4375" y="1532.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="333.4375" y="1546.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">A₀</text>
<rect x="405.75" y="1517.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="417.8125" y="1532.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="417.8125" y="1546.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">B₀</text>
<rect x="518.25" y="1517.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="530.3125" y="1532.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="530.3125" y="1546.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">A₁</text>
<rect x="630.75" y="1517.0" width="24.125" height="39" rx="5" fill="#ffedd5" stroke="#ea580c" />
<text x="642.8125" y="1532.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<text x="642.8125" y="1546.0" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">B₁</text>
<text x="797.5" y="1543.0" font-size="12" fill="#ea580c" font-weight="400" text-anchor="start">← 获得 row max 后可与 exp 重叠</text>
<text x="50" y="1621.0" font-size="16" fill="#0f172a" font-weight="700" text-anchor="start">TMA · w14 发请求</text>
<rect x="235" y="1597.0" width="900" height="39" rx="4" fill="#f8fafc" stroke="#e2e8f0" />
<rect x="237.0" y="1597.0" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="263.125" y="1620.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 K₁</text>
<rect x="293.25" y="1597.0" width="52.25" height="39" rx="5" fill="url(#hatch)" stroke="#dc2626" stroke-dasharray="5 4" />
<text x="319.375" y="1620.5" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">等槽</text>
<rect x="349.5" y="1597.0" width="52.25" height="39" rx="5" fill="#dcfce7" stroke="#16a34a" />
<text x="375.625" y="1620.5" font-size="12" fill="#16a34a" font-weight="700" text-anchor="middle">预取 V₁</text>
<text x="488.125" y="1623.0" font-size="12" fill="#334155" font-weight="400" text-anchor="start">← K₀ 已消费，原 slot 才能覆盖成 V₁</text>
<rect x="1160" y="1272.0" width="310" height="283" rx="7" fill="#f8fafc" stroke="#cbd5e1" />
<text x="1176" y="1301.0" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">改善：O 修正与 exp 重叠</text>
<text x="1176" y="1333.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">矩阵结果写 TMEM，其他角色接手。</text>
<text x="1176" y="1360.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">O 修正可与 softmax 的 exp 重叠。</text>
<text x="1176" y="1387.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">PV 仍要等 P 和 O 修正就绪。</text>
<text x="1176" y="1414.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">剩余 blocker：仍可能等 P 产出。</text>
<text x="1176" y="1441.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">8个 softmax warp 共用 MUFU；</text>
<text x="1176" y="1468.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">加 warp 不会增加指数单元带宽。</text>
<text x="1176" y="1495.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">本图保持 exp / O 修正工作量；</text>
<text x="1176" y="1522.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">不计 FMA exp、减少重缩放的收益。</text>
<text x="50" y="1694.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">黄色关键段错开：争用指数单元时同时做 exp 不增加总吞吐；softmax 的准备等其他阶段仍可重叠。</text>
<text x="50" y="1723.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">TMA 也可能等 slot，或计算可能等搬运完成。更多角色不会消除数据依赖和存储容量限制。</text>
<rect x="30" y="1767.0" width="1460" height="235" rx="7" fill="#f8fafc" stroke="#cbd5e1" />
<text x="50" y="1800.0" font-size="23" fill="#0f172a" font-weight="700" text-anchor="start">最关键的变化：同一条计算依赖，被安排到不同资源上推进</text>
<text x="50" y="1833.0" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">FA2</text>
<text x="160" y="1833.0" font-size="17" fill="#334155" font-weight="400" text-anchor="start">同批 warps：QK → softmax / O 修正 → PV；中间容易缺少可发的矩阵工作。</text>
<text x="50" y="1872.0" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">FA3</text>
<text x="160" y="1872.0" font-size="17" fill="#334155" font-weight="400" text-anchor="start">两个 WG 交错：一个做 softmax，另一个做矩阵乘；寄存器承担在途结果。</text>
<text x="50" y="1911.0" font-size="18" fill="#0f172a" font-weight="700" text-anchor="start">FA4</text>
<text x="160" y="1911.0" font-size="17" fill="#334155" font-weight="400" text-anchor="start">TMEM 让角色交接：矩阵乘、softmax、O 修正重叠；新的关键等待常是 P 的产出。</text>
<text x="50" y="1955.0" font-size="17" fill="#dc2626" font-weight="700" text-anchor="start">不变的约束：KV 就绪 → QK / PV；S 就绪 → softmax；P 与 O 就绪 → PV；消费完 → slot 可复用。</text>
<text x="30" y="2037.0" font-size="15" fill="#334155" font-weight="400" text-anchor="start">搬运行表示异步传输，warp 仅发请求；来源：FA3 overlap 发布说明、FA4 论文 §3.1。本图不是 profiler trace。</text>
<a href="https://pytorch.org/blog/flashattention-3/">
<text x="30" y="2067.0" font-size="15" fill="#2563eb" font-weight="400" text-anchor="start">FA3：pytorch.org/blog/flashattention-3/</text>
</a>
<a href="https://arxiv.org/html/2603.05451v1">
<text x="30" y="2095.0" font-size="15" fill="#2563eb" font-weight="400" text-anchor="start">FA4：arxiv.org/html/2603.05451v1</text>
</a>
</svg>
```


## 完整绘图生成器

```python
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET

ROOT = Path('/tmp/fa4_learn')
W,H=1520,2115
PX,UNIT=235,28.125
parts=[]
PALETTE={'mma':('#dbeafe','#2563eb'),'soft':('#fef3c7','#b45309'),
         'corr':('#ffedd5','#ea580c'),'copy':('#dcfce7','#16a34a')}

def txt(x,y,s,size=15,color='#334155',bold=False,anchor='start'):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{escape(s)}</text>')

def rect(x,y,w,h,fill='#ffffff',stroke='#cbd5e1',rx=7,dash=False):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"'+(' stroke-dasharray="5 4"' if dash else '')+'/>')

def line(x1,y1,x2,y2,color='#64748b',arrow=False,dash=False):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"'+(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="4 3"' if dash else '')+'/>')

def block(y,start,end,label,key,h=39,size=13):
    fill,stroke=PALETTE[key]
    x=PX+UNIT*start
    rect(x+2,y,UNIT*(end-start)-4,h,fill,stroke,5)
    if (end-start==1 or label.startswith('softmax')) and ' ' in label:
        first,second=label.split(' ',1)
        center=x+UNIT*(end-start)/2
        txt(center,y+15,first,9 if end-start==1 else 11,stroke,True,'middle')
        txt(center,y+29,second.replace(' → P ',' → P'),9 if end-start==1 else 10,stroke,True,'middle')
    else:
        txt(x+UNIT*(end-start)/2,y+h/2+4,label,min(size,10 if end-start==1 else 12),stroke,True,'middle')

def wait(y,start,end,label='',h=39):
    x=PX+UNIT*start
    rect(x+2,y,UNIT*(end-start)-4,h,'url(#hatch)','#dc2626',5,True)
    if label: txt(x+UNIT*(end-start)/2,y+h/2+4,label,10 if end-start==1 else 9 if end-start==2 else 12,'#dc2626',True,'middle')

def row(y,label):
    txt(50,y+24,label,16,'#0f172a',True)
    rect(PX,y,900,39,'#f8fafc','#e2e8f0',4)

def panel(y,h,title,subtitle):
    rect(30,y,1460,h,'#ffffff')
    txt(50,y+33,title,23,'#0f172a',True)
    txt(50,y+61,subtitle,15)
    line(PX,y+83,PX+900,y+83,arrow=True)
    txt(PX+900,y+77,'执行推进 →',12,'#64748b',anchor='end')

def notes(y,heading,lines):
    rect(1160,y,310,len(lines)*27+67,'#f8fafc')
    txt(1176,y+29,heading,18,'#0f172a',True)
    for i,s in enumerate(lines): txt(1176,y+61+27*i,s,15)

parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Noto Sans CJK SC,Arial,sans-serif" role="img">')
parts.append('<title>FA2、FA3、FA4：pipeline 如何缓解阻塞，哪些等待仍然存在</title><desc>按逻辑时间展示计算、softmax、O修正和异步搬运。红色斜纹标出等待P、等待指数单元或存储槽的依赖。图是机制示意，不是实测时间线。</desc>')
parts.append('<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#fef2f2"/><path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#fecaca" stroke-width="1"/></pattern></defs>')
rect(0,0,W,H,'#ffffff','#ffffff',0)
txt(30,42,'FA2 → FA3 → FA4：pipeline 改善了哪里的 blocker？',29,'#0f172a',True)
txt(30,72,'三图均处理同一组逻辑 A/B 分片 × 两轮 KV：每图均包含4份 QK / softmax / O 修正 / PV 工作。',16)
txt(30,99,'同类操作、相同工作量均同宽；不计硬件吞吐差异或软件减少运算量的收益。不同操作间的相对宽度仅作示意。',15)
for x,key,label in [(30,'mma','QK / PV · Tensor Core'),(350,'soft','softmax · 指数 / ALU'),(660,'corr','O 修正'),(835,'copy','异步搬运')]:
    fill,stroke=PALETTE[key];rect(x,113,18,18,fill,stroke,3);txt(x+25,128,label,14)
rect(1070,113,18,18,'url(#hatch)','#dc2626',3);txt(1095,128,'等待 / 资源竞争',14,'#dc2626')

panel(155,477,'FA2：同一批 warps 做搬运和整套计算','按 CTA 的计算阶段展开 A/B 两份工作；三图首轮 Q/K/V 均已就绪，启动和收尾省略。')
row(260,'warps · Q 分片 A')
row(340,'warps · Q 分片 B')
fa2_ops=[]
for r in range(2):
    base=16*r
    for y,which,ops in [
        (260,'A',[(0,2,'QK','mma'),(4,7,'softmax','soft'),(10,11,'O','corr'),(12,14,'PV','mma')]),
        (340,'B',[(2,4,'QK','mma'),(7,10,'softmax','soft'),(11,12,'O','corr'),(14,16,'PV','mma')])]:
        for a,b,name,key in ops:
            label=f'{name} {which}{r}'
            block(y,base+a,base+b,label,key)
            if key=='mma':fa2_ops.append((base+a,base+b,label))
row(420,'本 CTA 的矩阵工作')
for start,end,label in fa2_ops:block(420,start,end,label,'mma')
wait(420,4,12,'softmax / O 修正阶段：等 P₀')
wait(420,20,28,'softmax / O 修正阶段：等 P₁')
row(500,'异步搬运')
for start,end,label in [(4,6,'预取 K₁'),(16,18,'预取 V₁')]:block(500,start,end,label,'copy')
notes(256,'blocker：下一次 PV 缺输入',[
    'QK 算完不等于 PV 可以开始。',
    '必须先得到 softmax 的 P，',
    '并完成对应 O 的重缩放。',
    '同批 warps 此时做非矩阵运算。',
    '红色是本 CTA 的供给空档；',
    '其他 CTA 可填补，不等于全 GPU 闲置。'])
txt(50,588,'cp.async 已能重叠部分搬运；改进方向是让一份 Q 分片的非矩阵阶段，与另一份的矩阵工作交错。',15)
txt(50,616,'红色展示阶段式推进的供给空档；不表示 FA2 硬件禁止不同 warp 或不同 CTA 重叠计算。',15)

panel(577,490,'FA3：两个 consumer WG 交错，用另一组的矩阵工作填空档','每组仍负责 QK、softmax、O 修正、PV；此图主要展开组间交错，FA3 也支持组内重叠。')
row(682,'WG1 · Q 分片 A')
for a,b,s,k in [(0,2,'QK A₀','mma'),(2,5,'softmax A₀','soft'),(5,6,'修正','corr'),(6,8,'PV A₀','mma'),(8,10,'QK A₁','mma'),(10,13,'softmax A₁','soft'),(13,14,'修正','corr'),(14,16,'PV A₁','mma')]:block(682,a,b,s,k)
row(762,'WG2 · Q 分片 B')
for a,b,s,k in [(2,4,'QK B₀','mma'),(6,9,'softmax B₀','soft'),(9,10,'修正','corr'),(10,12,'PV B₀','mma'),(12,14,'QK B₁','mma'),(14,17,'softmax B₁','soft'),(17,18,'修正','corr'),(18,20,'PV B₁','mma')]:block(762,a,b,s,k)
wait(762,4,6,'等调度 / 依赖')
row(842,'本 CTA 的矩阵工作')
for a,b,s in [(0,2,'QK A₀'),(2,4,'QK B₀'),(6,8,'PV A₀'),(8,10,'QK A₁'),(10,12,'PV B₀'),(12,14,'QK B₁'),(14,16,'PV A₁'),(18,20,'PV B₁')]:block(842,a,b,s,'mma')
wait(842,4,6,'等 P / 修正')
wait(842,16,18,'等 P / 修正')
line(PX+3*UNIT,723,PX+3*UNIT,839,'#64748b',False,True)
txt(PX+3*UNIT+10,819,'WG1 softmax ∥ WG2 QK',12,'#0d9488')
row(922,'TMA · w0 发请求')
block(922,0,2,'预取 K₁','copy')
block(922,2,4,'预取 V₁','copy')
notes(677,'改善：用交错调度隐藏等待',[
    'WG1 做 softmax 时，',
    'WG2 可以发矩阵工作。',
    'TMA 搬运也独立推进。',
    '仍要等各组自己的 P / O 就绪。',
    '组内进一步重叠，需要同时',
    '持有更多 S / P / O 寄存器。',
    '寄存器压力限制流水线深度。'])
txt(50,1015,'WGMMA 是异步的；“每组做完整流程”不代表硬件要求该组所有操作严格串行。',15)
txt(50,1044,'更多重叠仍需足够寄存器保存尚未消费的结果；下一数据不能覆盖还在使用的缓冲。',15)

panel(1092,570,'FA4：TMEM 交接结果，矩阵乘、softmax、O 修正分给不同角色','矩阵、softmax、O 修正工作量与前两图一致；只展开角色解耦带来的调度变化。')
row(1197,'w12 / Tensor Core')
for a,b,s in [(0,2,'QK A₀'),(2,4,'QK B₀'),(5,7,'PV A₀'),(7,9,'QK A₁'),(9,11,'PV B₀'),(11,13,'QK B₁'),(13,15,'PV A₁'),(16,18,'PV B₁')]:block(1197,a,b,s,'mma')
for a,b in [(4,5),(15,16)]:wait(1197,a,b,'P',h=39)
row(1277,'softmax0 · A')
block(1277,2,5,'softmax A₀ → P A₀','soft')
block(1277,9,12,'softmax A₁ → P A₁','soft')
row(1357,'softmax1 · B')
wait(1357,4,5,'EXP')
block(1357,5,8,'softmax B₀ → P B₀','soft')
block(1357,13,16,'softmax B₁ → P B₁','soft')
line(PX+5*UNIT,1274,PX+5*UNIT,1239,'#dc2626',True)
txt(PX+5*UNIT+8,1256,'P A₀ 就绪 → PV A₀',12,'#dc2626')
row(1437,'correction · O 修正')
for a,b,s in [(3,4,'O A₀'),(6,7,'O B₀'),(10,11,'O A₁'),(14,15,'O B₁')]:block(1437,a,b,s,'corr',size=11)
txt(PX+20*UNIT,1463,'← 获得 row max 后可与 exp 重叠',12,'#ea580c')
row(1517,'TMA · w14 发请求')
block(1517,0,2,'预取 K₁','copy')
wait(1517,2,4,'等槽')
block(1517,4,6,'预取 V₁','copy')
txt(PX+9*UNIT,1543,'← K₀ 已消费，原 slot 才能覆盖成 V₁',12)
notes(1192,'改善：O 修正与 exp 重叠',[
    '矩阵结果写 TMEM，其他角色接手。',
    'O 修正可与 softmax 的 exp 重叠。',
    'PV 仍要等 P 和 O 修正就绪。',
    '剩余 blocker：仍可能等 P 产出。',
    '8个 softmax warp 共用 MUFU；',
    '加 warp 不会增加指数单元带宽。',
    '本图保持 exp / O 修正工作量；',
    '不计 FMA exp、减少重缩放的收益。'])
txt(50,1614,'黄色关键段错开：争用指数单元时同时做 exp 不增加总吞吐；softmax 的准备等其他阶段仍可重叠。',15)
txt(50,1643,'TMA 也可能等 slot，或计算可能等搬运完成。更多角色不会消除数据依赖和存储容量限制。',15)

rect(30,1687,1460,235,'#f8fafc')
txt(50,1720,'最关键的变化：同一条计算依赖，被安排到不同资源上推进',23,'#0f172a',True)
txt(50,1753,'FA2',18,'#0f172a',True)
txt(160,1753,'同批 warps：QK → softmax / O 修正 → PV；中间容易缺少可发的矩阵工作。',17)
txt(50,1792,'FA3',18,'#0f172a',True)
txt(160,1792,'两个 WG 交错：一个做 softmax，另一个做矩阵乘；寄存器承担在途结果。',17)
txt(50,1831,'FA4',18,'#0f172a',True)
txt(160,1831,'TMEM 让角色交接：矩阵乘、softmax、O 修正重叠；新的关键等待常是 P 的产出。',17)
txt(50,1875,'不变的约束：KV 就绪 → QK / PV；S 就绪 → softmax；P 与 O 就绪 → PV；消费完 → slot 可复用。',17,'#dc2626',True)
txt(30,1957,'搬运行表示异步传输，warp 仅发请求；来源：FA3 overlap 发布说明、FA4 论文 §3.1。本图不是 profiler trace。',15)
for y,url,label in [(1987,'https://pytorch.org/blog/flashattention-3/','FA3：pytorch.org/blog/flashattention-3/'),(2015,'https://arxiv.org/html/2603.05451v1','FA4：arxiv.org/html/2603.05451v1')]:
    parts.append(f'<a href="{url}">');txt(30,y,label,15,'#2563eb');parts.append('</a>')
parts.append('</svg>')
ET.register_namespace('', 'http://www.w3.org/2000/svg')
root=ET.fromstring('\n'.join(parts))
for element in root.iter():
    if element.tag.endswith('rect') and element.get('x')=='0' and element.get('y')=='0':
        element.set('height',str(H))
shift_following=False
for child in list(root):
    if child.tag.endswith('rect') and child.get('x')=='30' and child.get('y')=='577':
        shift_following=True
    if not shift_following:
        continue
    for element in child.iter():
        if element.tag.rsplit('}',1)[-1] in ['text','rect','line']:
            for attr in ['y','y1','y2']:
                if attr in element.attrib and float(element.attrib[attr])>=577:
                    element.set(attr,str(float(element.attrib[attr])+80))
for filename in ['fa2_fa3_fa4_pipeline_blockers.svg','fa2_fa3_fa4_pipeline_same_work.svg','fa2_fa3_fa4_pipeline_equal_op_work.svg']:
    ET.ElementTree(root).write(ROOT/filename,encoding='unicode')
print('Generated FA pipeline blockers diagram')
```


## 完整 PNG renderer

```python
import ctypes as c
import sys
from pathlib import Path
R=c.CDLL('librsvg-2.so.2'); C=c.CDLL('libcairo.so.2'); G=c.CDLL('libgobject-2.0.so.0')
R.rsvg_handle_new_from_file.argtypes=[c.c_char_p,c.POINTER(c.c_void_p)];R.rsvg_handle_new_from_file.restype=c.c_void_p
class Rect(c.Structure):
 _fields_=[('x',c.c_double),('y',c.c_double),('width',c.c_double),('height',c.c_double)]
R.rsvg_handle_render_document.argtypes=[c.c_void_p,c.c_void_p,c.POINTER(Rect),c.POINTER(c.c_void_p)];R.rsvg_handle_render_document.restype=c.c_int
C.cairo_image_surface_create.argtypes=[c.c_int,c.c_int,c.c_int];C.cairo_image_surface_create.restype=c.c_void_p
C.cairo_create.argtypes=[c.c_void_p];C.cairo_create.restype=c.c_void_p
C.cairo_scale.argtypes=[c.c_void_p,c.c_double,c.c_double]
C.cairo_surface_write_to_png.argtypes=[c.c_void_p,c.c_char_p];C.cairo_surface_write_to_png.restype=c.c_int
C.cairo_destroy.argtypes=[c.c_void_p];C.cairo_surface_destroy.argtypes=[c.c_void_p];G.g_object_unref.argtypes=[c.c_void_p]
import xml.etree.ElementTree as ET
p=Path(sys.argv[1]);root=ET.parse(p).getroot();w=int(root.attrib['width']);h=int(root.attrib['height'])
err=c.c_void_p();handle=R.rsvg_handle_new_from_file(str(p).encode(),c.byref(err));assert handle and not err.value
surface=C.cairo_image_surface_create(0,w*2,h*2);ctx=C.cairo_create(surface);C.cairo_scale(ctx,2,2)
assert R.rsvg_handle_render_document(handle,ctx,c.byref(Rect(0,0,w,h)),c.byref(err))
assert C.cairo_surface_write_to_png(surface,sys.argv[2].encode())==0
C.cairo_destroy(ctx);C.cairo_surface_destroy(surface);G.g_object_unref(handle)
print('Rendered PNG',w*2,h*2)
```
