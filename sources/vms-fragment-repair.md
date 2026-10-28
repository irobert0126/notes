# VMS 修复 fragment：2×2 示意图

这张图用四步说明：一段连续的 VMM 段中间释放后，VA 上出现空洞（fragment）；开启 VMS 后，空闲 chunk 被映射成一段新的连续 VA，请求由新段满足，不需要新分配。

新 session 只需读本 paste。

- 对应文件：本图源文件，内容见第 8 节

## 核心论点

**一句话目的：** 用实际例子解释 VMS 是怎么解决 fragment 的。

**核心知识点：** 用一个实际在 VA 上产生 fragment 的例子：VMS 把 fragment 区域（空洞）对应的 chunk 重新映射到段尾新的连续 VA 上，让这些 chunk 被使用，从而变相解决了"fragment 区域占着 chunk"的问题。VA 内部的 fragment 仍然存在，但 VA 可以比 HBM 大（预留 VA 不占显存，只有映射上的 chunk 占显存），这些 fragment 本质上不影响 HBM 的利用率。

**整张图的精髓：** fragment 的根源是"连续 VA 的要求"和"物理 chunk 不连续"之间的矛盾；VMS 不挪动已有的 chunk，而是再用 VMM 映射一次，绕开这个矛盾。

**措辞说明：** 原话里的"link"改为"映射"（VMM 的映射关系）。"fragment 区域占着 HBM trunk"更准确的说法是：空洞的旧映射仍然指向这些 chunk，但 chunk 本身已经空闲；VMS 让它们被新段 Y 使用，所以 HBM 的利用率没有损失。"VA 可以比 HBM 更长"改为"VA 可以比 HBM 大"，映射的总量仍然受 HBM 限制。

## 术语

| 术语 | 含义 |
|---|---|
| VMM | CUDA 低级虚拟内存管理：先预留一段 VA，再把物理 chunk 映射进去 |
| VMS | 建在 VMM 之上的拼接策略：从空闲 chunk 中选块，映射成一段新的连续 VA |
| VA block | 一段连续的虚拟地址，图中用起点编号标注 |
| chunk | 物理显存中的一块，图中每格 100MB 为示意 |
| 映射（mapping） | VA block 到 chunk 的指向，图中用箭头表示 |
| 空洞 / fragment | 段中间释放后留在 VA 上、不能用于大请求的一段地址 |
| 旧映射 | 空洞的 VA 仍指向原来的 chunk，没有解除，图中为虚线 |

## 背景

- 起因：想弄清 VMS 为什么能缓解 fragment，所以把"VA 上的碎片"和"物理 chunk"分开画出来。
- 图是示意：块大小、编号、个数都不是实测。
- VMS 对应的论文：GMLake: Efficient and Transparent GPU Memory Defragmentation for Large-scale DNN Training with Virtual Memory Stitching（arXiv 2401.08156，ASPLOS'24）。

## 知识

| ID | 内容 | 来源 | 状态 | 确认 | 你的理解 | 对应图元 |
|---|---|---|---|---|---|---|
| K1 | VMM 把 VA 与物理 chunk 分开：VA 块通过映射指向 chunk，chunk 的物理位置可以散落。 | 会话中的 VMM 说明，CUDA 通用定义，未对照官方文档 | unverified | 已确认 |  | 段 X 与 HBM chunk 之间的映射箭头 |
| K2 | 一段连续 VA 可以映射到不连续的 chunk，VA 上仍然连续。 | 同 K1 | unverified | 已确认 |  | 图 1 |
| K3 | 释放段中间的一段后，VA 上留下空洞（fragment）；空洞的映射没有被自动撤销，需要显式解除。 | 会话中的讨论；解除映射需要显式调用（通用做法） | unverified | 已确认 |  | 图 2 空洞（虚线） |
| K4 | 同一个 chunk 可以同时被两段 VA 映射（旧映射与新映射并存）。 | 会话中的讨论 | unverified | 已确认 |  | 图 4 两条指向同一 chunk 的箭头 |
| K5 | VMS：从空闲 chunk 中选出若干块（不要求物理连续），映射到一段新的连续 VA；选块本身不新建 chunk。 | 会话中对 VMS 的说明，设计层面，未对照源码 | unverified | 已确认 |  | 图 3 选中；图 4 段 Y |
| K6 | 不新分配时 reserved 不变；需要新分配时 reserved 增长。 | 由定义推出 | inferred | 已确认 |  | 图 2、图 4 的 reserved 文字 |
| K7 | VA 预留不占显存，只有映射上的 chunk 占显存；映射的总量不能超过 HBM。 | 由定义推出 | inferred | 已确认 |  | VA 段方框 |
| K8 | VA 上的空洞本身不消耗 HBM：HBM 的利用率只取决于有没有 chunk 被映射使用；空洞的 VA 再多也不占显存。 | 由定义推出 | inferred | 已确认 |  | 图 2 空洞（虚线）；图 4 |

状态：verified（读过注释或做过算术）/ inferred（由定义或逻辑推出）/ unverified（通用机制或记忆，未核实）。

## 分组讲解

按推理链分两组。

### 第 1 组：VA 和 chunk 分开，fragment 从哪里来（K1、K2、K3、K4）

- **K1，VA 和 chunk 分开：** VA 是一段连续的地址，chunk 是显存里的物理块。映射把 VA 指向 chunk，所以 chunk 的位置可以散落。
- **K2，VA 连续与 chunk 位置无关：** 段 X 的 VA 连续，但它映射的 6 个 chunk 在显存里不相邻。
- **K3，释放中间一段留下空洞：** 释放中间 200MB 后，那段 VA 成为空洞。空洞的映射还在，它的 chunk 变成空闲，但空洞的地址不能拿来放连续的大请求。后面的段 X3 的 VA 不会因此移动。
- **K4，同一 chunk 可以被两段映射：** 空洞的映射没有撤销时，同一个 chunk 可以同时出现在空洞和之后的新段里（见第 2 组）。

### 第 2 组：VMS 怎么修复（K5、K6、K7、K8）

- **K5，VMS 选块再映射：** 请求 300MB 时，VMS 从空洞的 2 个 chunk 和空闲的 1 个 chunk 中选块，映射到一段新的连续 VA（段 Y）。选块本身不新建 chunk。
- **K6，reserved 的变化：** 没有新分配时 reserved 不变；不开 VMS 时只能新分配，reserved 增长。
- **K7，VA 与显存的关系：** VA 预留不占显存，所以 VA 地址空间可以比显存大；但映射的 chunk 总量不能超过 HBM。
- **K8，fragment 不等于 HBM 浪费：** VA 上的空洞只是地址，不消耗显存；HBM 有没有浪费只看 chunk 是否被映射使用。VA 可以比 HBM 大，但映射的总量仍不能超过 HBM（K7）。

## 图的内容

按从左到右、从上到下的顺序看：

1. **图 1，一段连续的 VMM 段（VMS 未开）：** 段 X 是 3 个 200MB 的 VA block，起点分别是 000、002、004，每个 block 映射 2 个 chunk。chunk 在 HBM 里散落，009 那格是 100MB 空闲。
2. **图 2，中间释放，产生 fragment（VMS 未开）：** block 002（200MB）释放，留下空洞；空洞的旧映射仍指向 chunk 004、005（虚线）。空闲 300MB，但连续 VA 最大只有 200MB。没有 VMS 时，300MB 的请求只能新分配。
3. **图 3，开启 VMS，选出可复用的空闲 chunk：** 请求 300MB，VMS 选中 chunk 004、005、009。这一步只选块，还没有映射。
4. **图 4，VMM 映射成新的连续 VA：** 段 Y（block 010 起，300MB，连续）映射 chunk 004、005、009。这三个 chunk 在显存里不相邻；004、005 与空洞共用。没有新分配，reserved 不变。

图 4 的标题"fragment 修复"指的是：空洞的 VA 仍在，物理空闲被 Y 复用，不需要新分配。

## 结构

- 布局：2×2 四个面板，依次为图 1 → 图 2 → 图 3 → 图 4。
- 实体：VMM 段框（VA block，框内写名字和状态，起点编号在框外）；HBM chunk 框（每格 100MB，编号在框下）。
- 关系：实线箭头 = 有效映射；虚线箭头 = 旧映射（空洞仍指向的 chunk）。
- 地址标注：起点编号和 chunk 编号都在框外，用竖直细虚线连到框。
- 颜色：绿 = VA 段；紫 = 映射到段的 chunk；灰 = 空闲；深灰 = 其他段占用；橙 = 选中；虚线 = 旧映射。
- 阅读顺序：左上 → 右上 → 左下 → 右下。

## SVG 约定

- 尺寸：`viewBox="0 0 1240 760"`
- 字体：`'PingFang SC', 'Hiragino Sans GB', 'Helvetica Neue', Arial`
- 编号：VA block 起点和 chunk 编号都是 dummy（从 000 起），不是真实地址。

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="760" viewBox="0 0 1240 760" font-family="'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#333333"/></marker></defs>
<rect width="1240" height="760" fill="#ffffff"/>
<rect x="10" y="10" width="600" height="360" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="26" y="36" font-size="14" font-weight="bold" fill="#222222">图 1　一段连续的 VMM 段（VMS 未开）</text>
<text x="594" y="36" font-size="11" font-weight="bold" text-anchor="end" fill="#666666">VMS：关</text>
<text x="26" y="64" font-size="11" font-weight="bold" fill="#3d2466">VMM（虚拟地址，块起点编号，每格 100MB）</text>
<rect x="26" y="90" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="81.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X1 200MB</text>
<text x="81.0" y="125" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="26" y1="80" x2="26" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="26" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="138" y="90" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="193.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X2 200MB</text>
<text x="193.0" y="125" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="138" y1="80" x2="138" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="138" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="250" y="90" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="305.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X3 200MB</text>
<text x="305.0" y="125" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="250" y1="80" x2="250" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="250" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="26" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="52.0" y1="212" x2="52.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="52.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="82" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="108.0" y1="212" x2="108.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="108.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">001</text>
<rect x="138" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="164.0" y1="212" x2="164.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="164.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="194" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="220.0" y1="212" x2="220.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="220.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">003</text>
<rect x="250" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="276.0" y1="212" x2="276.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="276.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="306" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="332.0" y1="212" x2="332.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="332.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">005</text>
<rect x="362" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="388.0" y1="212" x2="388.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="388.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">006</text>
<rect x="418" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="444.0" y1="212" x2="444.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="444.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">007</text>
<rect x="474" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="500.0" y1="212" x2="500.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="500.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">008</text>
<rect x="530" y="178" width="52" height="34" rx="3" fill="#e0e0e0" stroke="#999999" stroke-width="1.3"/>
<line x1="556.0" y1="212" x2="556.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="556.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">009</text>
<text x="26" y="256" font-size="11" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 100MB，编号为示意）</text>
<line x1="81.0" y1="138" x2="52.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="81.0" y1="138" x2="164.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="193.0" y1="138" x2="276.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="193.0" y1="138" x2="332.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="305.0" y1="138" x2="444.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="305.0" y1="138" x2="500.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<text x="26" y="280" font-size="12" fill="#222222">段 X：000 起，3 个 VA block 连续；每个 block 映射 2 个 chunk。</text>
<text x="26" y="298" font-size="12" fill="#222222">chunk 的编号是散落的；009 那格 100MB 空闲没有用上。</text>
<text x="26" y="356" font-size="10" fill="#666666">绿 = VA 段；紫 = 映射到段的 chunk；灰 = 空闲；深灰 = 其他段占用；橙 = 选中；虚线 = 旧映射</text>
<rect x="630" y="10" width="600" height="360" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="646" y="36" font-size="14" font-weight="bold" fill="#222222">图 2　中间释放 200MB，产生 fragment（VMS 未开）</text>
<text x="1214" y="36" font-size="11" font-weight="bold" text-anchor="end" fill="#666666">VMS：关</text>
<text x="646" y="64" font-size="11" font-weight="bold" fill="#3d2466">VMM（虚拟地址，块起点编号，每格 100MB）</text>
<rect x="646" y="90" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="701.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X1 200MB</text>
<text x="701.0" y="125" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="646" y1="80" x2="646" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="646" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="766" y="90" width="110" height="48" rx="3" fill="#ffffff" stroke="#922b21" stroke-width="1.3" stroke-dasharray="5 3"/>
<text x="821.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#922b21">空洞 200MB</text>
<text x="821.0" y="125" font-size="9" text-anchor="middle" fill="#922b21">旧映射仍在</text>
<line x1="766" y1="80" x2="766" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="766" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="886" y="90" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="941.0" y="109" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X3 200MB</text>
<text x="941.0" y="125" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="886" y1="80" x2="886" y2="90" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="886" y="76" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="646" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="672.0" y1="212" x2="672.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="672.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="702" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="728.0" y1="212" x2="728.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="728.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">001</text>
<rect x="758" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="784.0" y1="212" x2="784.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="784.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="814" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="840.0" y1="212" x2="840.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="840.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">003</text>
<rect x="870" y="178" width="52" height="34" rx="3" fill="#e0e0e0" stroke="#999999" stroke-width="1.3"/>
<line x1="896.0" y1="212" x2="896.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="896.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="926" y="178" width="52" height="34" rx="3" fill="#e0e0e0" stroke="#999999" stroke-width="1.3"/>
<line x1="952.0" y1="212" x2="952.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="952.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">005</text>
<rect x="982" y="178" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="1008.0" y1="212" x2="1008.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1008.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">006</text>
<rect x="1038" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="1064.0" y1="212" x2="1064.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1064.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">007</text>
<rect x="1094" y="178" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="1120.0" y1="212" x2="1120.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1120.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">008</text>
<rect x="1150" y="178" width="52" height="34" rx="3" fill="#e0e0e0" stroke="#999999" stroke-width="1.3"/>
<line x1="1176.0" y1="212" x2="1176.0" y2="224" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1176.0" y="236" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">009</text>
<text x="646" y="256" font-size="11" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 100MB，编号为示意）</text>
<line x1="701.0" y1="138" x2="672.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="701.0" y1="138" x2="784.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="941.0" y1="138" x2="1064.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="941.0" y1="138" x2="1120.0" y2="178" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="821.0" y1="138" x2="896.0" y2="178" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<line x1="821.0" y1="138" x2="952.0" y2="178" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<text x="646" y="280" font-size="12" fill="#222222">中间 200MB 释放：block 002 仍在，旧映射没有撤销（虚线）。</text>
<text x="646" y="298" font-size="12" fill="#222222">它仍指向 chunk 004、005；空闲 300MB，但连续 VA 最大 200MB。</text>
<text x="646" y="316" font-size="12" fill="#222222">没有 VMS：300MB 的请求只能新分配，reserved 会增长。</text>
<text x="646" y="356" font-size="10" fill="#666666">绿 = VA 段；紫 = 映射到段的 chunk；灰 = 空闲；深灰 = 其他段占用；橙 = 选中；虚线 = 旧映射</text>
<rect x="10" y="390" width="600" height="360" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="26" y="416" font-size="14" font-weight="bold" fill="#222222">图 3　开启 VMS：选出可复用的空闲 chunk</text>
<text x="594" y="416" font-size="11" font-weight="bold" text-anchor="end" fill="#225c33">VMS：开</text>
<text x="26" y="444" font-size="11" font-weight="bold" fill="#3d2466">VMM（虚拟地址，块起点编号，每格 100MB）</text>
<rect x="26" y="470" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="81.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X1 200MB</text>
<text x="81.0" y="505" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="26" y1="460" x2="26" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="26" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="146" y="470" width="110" height="48" rx="3" fill="#ffffff" stroke="#b36b00" stroke-width="1.3" stroke-dasharray="5 3"/>
<text x="201.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#8a5300">空洞 200MB</text>
<text x="201.0" y="505" font-size="9" text-anchor="middle" fill="#8a5300">待复用</text>
<line x1="146" y1="460" x2="146" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="146" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="266" y="470" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="321.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X3 200MB</text>
<text x="321.0" y="505" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="266" y1="460" x2="266" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="266" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="26" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="52.0" y1="592" x2="52.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="52.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="82" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="108.0" y1="592" x2="108.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="108.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">001</text>
<rect x="138" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="164.0" y1="592" x2="164.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="164.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="194" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="220.0" y1="592" x2="220.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="220.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">003</text>
<rect x="250" y="558" width="52" height="34" rx="3" fill="#ffd27a" stroke="#b36b00" stroke-width="1.3"/>
<line x1="276.0" y1="592" x2="276.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="276.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="306" y="558" width="52" height="34" rx="3" fill="#ffd27a" stroke="#b36b00" stroke-width="1.3"/>
<line x1="332.0" y1="592" x2="332.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="332.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">005</text>
<rect x="362" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="388.0" y1="592" x2="388.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="388.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">006</text>
<rect x="418" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="444.0" y1="592" x2="444.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="444.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">007</text>
<rect x="474" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="500.0" y1="592" x2="500.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="500.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">008</text>
<rect x="530" y="558" width="52" height="34" rx="3" fill="#ffd27a" stroke="#b36b00" stroke-width="1.3"/>
<line x1="556.0" y1="592" x2="556.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="556.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">009</text>
<text x="26" y="636" font-size="11" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 100MB，编号为示意）</text>
<line x1="81.0" y1="518" x2="52.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="81.0" y1="518" x2="164.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="321.0" y1="518" x2="444.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="321.0" y1="518" x2="500.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="201.0" y1="518" x2="276.0" y2="558" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<line x1="201.0" y1="518" x2="332.0" y2="558" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<text x="26" y="660" font-size="12" fill="#222222">请求 300MB：VMS 选中 chunk 004、005、009。</text>
<text x="26" y="678" font-size="12" fill="#222222">这一步只选块，还没有映射，也不新建 chunk。</text>
<text x="26" y="736" font-size="10" fill="#666666">绿 = VA 段；紫 = 映射到段的 chunk；灰 = 空闲；深灰 = 其他段占用；橙 = 选中；虚线 = 旧映射</text>
<rect x="630" y="390" width="600" height="360" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="646" y="416" font-size="14" font-weight="bold" fill="#222222">图 4　VMM 映射成新的连续 VA，fragment 修复</text>
<text x="1214" y="416" font-size="11" font-weight="bold" text-anchor="end" fill="#225c33">VMS：开</text>
<text x="646" y="444" font-size="11" font-weight="bold" fill="#3d2466">VMM（虚拟地址，块起点编号，每格 100MB）</text>
<rect x="646" y="470" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="701.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X1 200MB</text>
<text x="701.0" y="505" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="646" y1="460" x2="646" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="646" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="766" y="470" width="110" height="48" rx="3" fill="#ffffff" stroke="#922b21" stroke-width="1.3" stroke-dasharray="5 3"/>
<text x="821.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#922b21">空洞 200MB</text>
<text x="821.0" y="505" font-size="9" text-anchor="middle" fill="#922b21">旧映射仍在</text>
<line x1="766" y1="460" x2="766" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="766" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="886" y="470" width="110" height="48" rx="3" fill="#7fcf94" stroke="#225c33" stroke-width="1.3"/>
<text x="941.0" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#173f25">X3 200MB</text>
<text x="941.0" y="505" font-size="9" text-anchor="middle" fill="#173f25">段 X</text>
<line x1="886" y1="460" x2="886" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="886" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="1006" y="470" width="165" height="48" rx="3" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
<text x="1088.5" y="489" font-size="10" font-weight="bold" text-anchor="middle" fill="#0e6655">Y 300MB</text>
<text x="1088.5" y="505" font-size="9" text-anchor="middle" fill="#0e6655">新段，连续</text>
<line x1="1006" y1="460" x2="1006" y2="470" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1006" y="456" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">010</text>
<rect x="646" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="672.0" y1="592" x2="672.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="672.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">000</text>
<rect x="702" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="728.0" y1="592" x2="728.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="728.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">001</text>
<rect x="758" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="784.0" y1="592" x2="784.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="784.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">002</text>
<rect x="814" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="840.0" y1="592" x2="840.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="840.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">003</text>
<rect x="870" y="558" width="52" height="34" rx="3" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
<line x1="896.0" y1="592" x2="896.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="896.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">004</text>
<rect x="926" y="558" width="52" height="34" rx="3" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
<line x1="952.0" y1="592" x2="952.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="952.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">005</text>
<rect x="982" y="558" width="52" height="34" rx="3" fill="#7a7a7a" stroke="#4d4d4d" stroke-width="1.3"/>
<line x1="1008.0" y1="592" x2="1008.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1008.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">006</text>
<rect x="1038" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="1064.0" y1="592" x2="1064.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1064.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">007</text>
<rect x="1094" y="558" width="52" height="34" rx="3" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
<line x1="1120.0" y1="592" x2="1120.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1120.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">008</text>
<rect x="1150" y="558" width="52" height="34" rx="3" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
<line x1="1176.0" y1="592" x2="1176.0" y2="604" stroke="#999999" stroke-width="1" stroke-dasharray="2 2"/>
<text x="1176.0" y="616" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#555555">009</text>
<text x="646" y="636" font-size="11" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 100MB，编号为示意）</text>
<line x1="701.0" y1="518" x2="672.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="701.0" y1="518" x2="784.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="941.0" y1="518" x2="1064.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="941.0" y1="518" x2="1120.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="821.0" y1="518" x2="896.0" y2="558" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<line x1="821.0" y1="518" x2="952.0" y2="558" stroke="#666666" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#arr)"/>
<line x1="1088.5" y1="518" x2="1176.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="1088.5" y1="518" x2="896.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<line x1="1088.5" y1="518" x2="952.0" y2="558" stroke="#333333" stroke-width="1.1" marker-end="url(#arr)"/>
<text x="646" y="660" font-size="12" fill="#222222">新段 Y：block 010 起，300MB，连续。</text>
<text x="646" y="678" font-size="12" fill="#222222">它映射的 chunk 004、005、009 物理上不连续。</text>
<text x="646" y="696" font-size="12" fill="#222222">004、005 与空洞共用；没有新分配，reserved 不变。</text>
<text x="646" y="736" font-size="10" fill="#666666">绿 = VA 段；紫 = 映射到段的 chunk；灰 = 空闲；深灰 = 其他段占用；橙 = 选中；虚线 = 旧映射</text>
</svg>
```
