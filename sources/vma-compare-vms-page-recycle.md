# VA 映射减小 fragment：VMS 拼接 vs 页回收（compare_vms_page_recycle）

新 session 只需读本 paste。

- 基于：无（首版，取代 P2575698707）
- 对应文件：`compare_vms_page_recycle.svg`（源文件，内容同第 9 节）

## 核心论点

**一句话目的：** 比较两种VA地址映射方式减小fragment的方法。

**核心知识点（草稿，待你确认）：** 两种方法的根本区别在**回收的动作**。VMS 的回收是 GC：不在释放 tensor 时做，而是等触发条件（VA 分配失败、autoGC 超限、分配失败后的重试、emptyCache）之一满足，并且 cudaEvent 完成之后，才删掉旧 VA 的反向引用，再 cuMemUnmap；PhyBlock 保留，reserved 不降。页回收的回收是搬页：miss 时同步 unmap 空洞里的 idle 页，再 map 到段尾；每个页任何时刻只有一段 VA；回收发生在分配路径上，reserved 也不降。要分清这两种回收，需要熟悉 GC 的每一步和每个触发点，因为 VMS 的旧映射什么时候消失，完全由 GC 决定。

**整张图的精髓：** 同一个请求，VMS 加一条映射，旧映射留给 GC 回收；页回收移动映射，miss 时当场回收。

**措辞说明：** chunk 编号、地址、页回收的 cap（10 MB）都是示意。流程是按代码路径（VMS）和文档文字（页回收）手推的，没有实际运行。

## 术语

| 术语 | 含义 |
|---|---|
| VA 块 / VirDevPtr | 一段预留的 VA，映射若干 chunk |
| chunk / PhyBlock | 2 MB 的物理块（HBM 句柄），VA 通过映射指向它 |
| mapped_blocks（m） | 映射这个 chunk 的 VA 条数；m ≥ 2 表示同一 chunk 有多段 VA |
| free（chunk） | 分配器记账：这个 chunk 能否被取走，与映射无关 |
| allocated（Block） | 这个块是否被占用（在用） |
| 拼接（VMS） | 从空闲块取 chunk，建新的 VA 段映射它们；旧映射保留，由 GC 撤销 |
| 页回收（recycler） | miss 时同步 unmap 空洞里的 idle 页，再 map 到段尾 |
| GC（VMS） | 延迟撤销拼接块的 VA 映射；只在特定触发条件下运行，见 K8 |
| 空洞 | 段内释放后留下的 VA 区域；页回收里它可以是未映射的空格子 |
| reserved | 分配器持有的物理内存；只有物理句柄释放才会下降 |
| fragLimit（VMS） | 拼接的最小请求大小，默认 512 MB |
| cap（页回收） | 回收触发的 reserved 阈值（图里假设 10 MB） |

## 背景

- 起因：同一个碎片例子里，两种方法在页表上做了什么不同的操作，回收的动作和时机有什么不同。
- 例子：6 个 chunk（H0–H5）。H0–H2 释放，H3 被分配，H4–H5 空闲。请求 4 块（8 MB）：连续空闲最多 3 块，但空闲总共 5 块。

## 知识

| ID | 内容 | 来源 | 状态 | 确认 | 你的理解 | 对应图元 |
|---|---|---|---|---|---|---|
| K1 | 同一段连续 VA 可以映射不连续的 chunk；连续的是 VA，不是物理内存。 | CUDA VMM 通用概念；图里 VA 格子与 HBM 格子之间的箭头 | unverified（通用机制） | 待确认 |  | ③④ 各面板的箭头 |
| K2 | VMS 拼接：从空闲 Block 取 chunk，新建 VmmSegment(phy_blocks)，每个 chunk 一次 cuMemMap；旧映射保留。 | CUDACachingAllocator.cpp :2704–2989；cuda_vmm_allocator.h :351、:410–420 | verified（读过代码） | 待确认 |  | ④ V4 |
| K3 | VMS 拼接门槛：非 small pool，请求 ≥ fragLimit（默认 512 MB），且空闲总量 ≥ 请求；否则拼接失败。 | CUDACachingAllocator.cpp :2713、:2743、:2774 | verified（读过代码） | 待确认 |  | ③ 分叉点 |
| K4 | VMS 释放 tensor 不 unmap：update_block 只改 free、计数、池子。 | CUDACachingAllocator.cpp :1593–1660 | verified（读过代码） | 待确认 |  | ② 释放；⑤ T4 释放 |
| K5 | VMS 同一 chunk 可以有多条映射：mapped_blocks 是列表，拼接时追加一条，旧条目不删。 | cuda_vmm_allocator.h :264；CUDACachingAllocator.cpp :2956 | verified（读过代码） | 待确认 |  | ④ H0–H2、H4 的 m = 2 |
| K6 | VMS 的 unmap 只在 VirDevPtr 释放时发生（cuMemUnmap + cuMemAddressFree），整个 patch 里只有这一处 cuMemUnmap。 | cuda_vmm_allocator.h :300–306 | verified（读过代码） | 待确认 |  | ⑥ GC |
| K7 | VMS GC：对候选块先 cudaEventQuery，完成后才删 mapped_blocks 里的反向引用并释放 VA；未完成则停下。 | CUDACachingAllocator.cpp :2554–2580、:2613 | verified（读过代码） | 待确认 |  | ⑥ GC |
| K8 | VMS GC 的触发：VA 分配失败、autoGC 超限（默认 1000 GB）、分配失败后的后续重试、emptyCache。普通的释放和正常分配成功都不触发。 | CUDACachingAllocator.cpp :2909、:2980、:3175、:1755 | verified（读过代码） | 待确认 |  | ⑥ GC |
| K9 | VMS 新 VA 由驱动选地址：mapVirAddr 传入的 device_ptr = 0，没有地址提示。 | cuda_vmm_allocator.h :484–497 | verified（读过代码） | 待确认 |  | ④ V4 的位置 |
| K10 | 页回收：miss 时 recycler 同步 unmap 空洞里的 idle 页，再 map 到段尾；同一页任何时刻只有一段 VA。 | 文档原文（文字描述，未对照代码） | unverified（文档，未对照代码） | 待确认 |  | ④、⑤ 页回收支线 |
| K11 | 页回收：只在 miss 且 reserved 超过 cap 时才回收；释放 tensor 不 unmap。cap = 10 MB 是图里的假设。 | 文档原文；cap 数值是假设 | unverified（文档，未对照代码） | 待确认 |  | ③ cap；② 释放 |
| K12 | 页回收的实测：64 MiB 页，map 调用减少 40%；页移动撞 host-wide driver lock，是主要成本。 | 文档原文（文档自己的测量） | unverified（文档，未对照代码） | 待确认 |  | 区别表的代价行 |
| K13 | 两边都不降 reserved：unmap 和 GC 只删映射；reserved 下降要物理句柄释放（VMS：PhyBlock 析构时 cuMemRelease）。 | cuda_vmm_allocator.h :243、:248–253；文档里 empty_cache 的实测 | verified（VMS 部分） | 待确认 |  | 表中 reserved 行 |
| K14 | 两边的前提：空闲总量 ≥ 请求才能不新分配；不够时新建段，reserved 增长。VMS 默认路径不把空闲块和新块混用（reAlloc 默认 0）。 | CUDACachingAllocator.cpp :2774；realloc_block 开头的 reAlloc 默认值；文档文字 | verified（VMS 部分）；文档文字 | 待确认 |  | ③ 请求；两张支线的 ④ 之前 |
| K15 | 例子结果：VMS 拼接 H0–H2 + H4 成 V4，H5 切开放回空闲；页回收搬 H0、H1、H2、H4 到 S6–S9，H5 仍空闲。 | 手推（代码路径 / 文档文字），未运行 | inferred（手推） | 待确认 |  | ④ 两条支线 |
| K16 | 本例只说明页表操作，不说明性能。两边 driver 调用的次数和耗时需要实测（VMS 2 MB 粒度 vs 页回收 64 MiB 页）。 | 推断 | inferred（手推） | 待确认 |  | 区别表的 unmap 时机行 |
| K17 | VMS GC 逐块做的事：查 cudaEventQuery；从每个 chunk 的 mapped_blocks 删掉本块的条目；从 fragmented（time > 0 时还有 free_fused）池 erase；delete 块，VmmSegment 和 VirDevPtr 析构，cuMemUnmap + cuMemAddressFree。 | CUDACachingAllocator.cpp :2554–2600；cuda_vmm_allocator.h :300–306 | verified（读过代码） | 待确认 |  | ⑥ GC 的五步 |
| K18 | VMS GC 不做的事：不 cuMemRelease（PhyBlock 保留，reserved 不降）；不改 PhyBlock.free；不处理非拼接块（V_a、V2 这类）。 | cuda_vmm_allocator.h :243；CUDACachingAllocator.cpp GC 函数（末尾的统计部分我没有读完） | verified（读过代码，末尾未读） | 待确认 |  | ⑥ GC 之后的状态 |
| K19 | 页回收的回收动作：在 miss 路径上同步 unmap 再 map，不在释放 tensor 时做，也不靠 GC；页本身不释放 HBM。 | 文档原文（文字描述，未对照代码） | unverified（文档，未对照代码） | 待确认 |  | ④ 页回收的回收动作 |
| K20 | 两种回收的时间点不同：VMS 的旧映射要等 GC 触发并且 event 完成才撤销；页回收的搬页在 miss 当场完成。 | K7、K8、K10 推出 | inferred（手推） | 待确认 |  | 区别表的 unmap 时机行 |

状态：verified（读过 GMLake patch 的代码）/ inferred（手推的例子，未运行）/ unverified（通用机制，或文档原文未对照代码）。确认：待确认 / 已确认。

## 分组讲解（待确认）

### 第 1 组：两种方法怎么让空闲 chunk 变成连续 VA（K1–K5、K9–K10）

- **K1，连续的是 VA。** 一段 VA 可以映射不连续的 chunk，把空闲 chunk 拼起来的关键是 VA 上的映射。
- **VMS，加映射。** 拼接时新建一段 VA，映射取来的 chunk；原来的 VA 块和映射都保留（K2、K5）。同一个 chunk 因此有两段 VA。
- **页回收，移映射。** miss 时先从空洞上 unmap，再 map 到段尾（K10）。每个页在任何时刻只映射一次。

检查题：拼接之后 H1 同时被哪两段 VA 映射？这两段 VA 在记账上各是什么状态？

### 第 2 组：回收的动作和 GC（K6–K8、K17–K20）

- **VMS 的回收是 GC。** 它做五步：查 event、删反向引用、从池里 erase、析构 VA、cuMemUnmap（K17）。它不 cuMemRelease，不改 free，也不处理非拼接块（K18）。
- **GC 的触发不止一个。** VA 分配失败、autoGC 超限、分配失败后的重试、emptyCache（K8）。普通的释放和正常分配都不触发它。只要 event 没完成，GC 就停下来，留到下次（K7）。
- **页回收的回收是搬页。** 只在 miss 路径上同步做，释放 tensor 不做，也不靠 GC（K19）。
- **时间点不同。** VMS 的旧映射要等 GC 触发、event 完成才撤销；页回收的搬页在 miss 当场完成（K20）。
- **两边都不降 reserved**（K13）。

检查题：H0 的旧映射（V_a 的那一段）在哪些条件下会被 GC 撤销？如果 event 没完成，这一轮 GC 怎么处理它？页回收的 H0 在 unmap 之后还占 HBM 吗？

## 图的内容

1. **共同部分（①②③）。** 初始 H0–H2 在用，H3 和 H4–H5 空闲；释放 H0–H2；H3 被分配；请求 4 块，分叉点在 ③。
2. **VMS 支线（④⑤⑥）。** ④ 拼接 H0–H2 和 H4，新建 V4，H5 切开放回；⑤ T4 释放，映射都还在；⑥ GC 撤销 V4 的映射。
3. **页回收支线（④⑤⑥）。** ④ unmap H0、H1、H2、H4 的空洞页；⑤ map 到段尾 S6–S9；⑥ T4 释放，页仍映射在 S6–S9。
4. **区别表。** 第 ④ 步做什么、T4 的 VA 位置、空洞的 VA 之后怎样、每块 HBM 的映射数、物理页的 VA、unmap 的时机、释放之后的状态、reserved。

## 结构

- 布局：三行（共同 / VMS / 页回收），底部是区别表。每个分镜上半是 VA 格子，下半是 HBM chunk，最下面列出变化的 flag。
- 颜色：绿 = 在用；灰 = 空闲；虚线橙 = 映射还在，但记账是别的状态；橙框 = 同一 chunk 被两段 VA 映射；虚线方框 = 未分配。

## SVG 约定

- 尺寸：`viewBox="0 0 1240 1536"`
- 字体：`'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial`
- chunk 编号 H0–H5 和地址是示意；flag 名（free、m、mapped、released、allocated、idle、owner）按上面术语表的定义。

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1536" viewBox="0 0 1240 1536" font-family="'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#34495e"/></marker></defs>
<rect width="1240" height="1536" fill="#ffffff"/>
<text x="10.0" y="28.0" font-size="16" font-weight="bold" fill="#1c2833">VMS 与 页回收：同一个例子，前三步相同，从第 ④ 步起分叉</text>
<text x="10.0" y="50.0" font-size="11" fill="#566573">例子：6 个 chunk（H0–H5）。H0–H2 释放，H3 分配，H4–H5 空闲；请求 4 块（8 MB）。chunk / 页 = 2 MB。</text>
<rect x="10.0" y="64.0" width="1220.0" height="22.0" rx="4" fill="#5d6d7e" stroke="none" stroke-width="1"/>
<text x="20.0" y="79.0" font-size="11" font-weight="bold" fill="#ffffff">共同的部分：①②③ 两边一样（HBM 块、张量、释放、分配都相同；请求在 ③ 之后分叉）</text>
<rect x="10.0" y="92.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="22.0" y="114.0" font-size="13" font-weight="bold" fill="#1c2833">① 初始：H0–H2 在用，H3–H5 空闲</text>
<text x="22.0" y="130.0" font-size="10" fill="#33414e">A（T_A）映射 H0–H2，在用，6 MB。H3 和 C（H4–H5）是空闲的，它们的映射都还</text>
<text x="22.0" y="144.0" font-size="10" fill="#33414e">在。</text>
<text x="22.0" y="176.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="18.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="34.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="34.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="52.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="68.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="86.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="102.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="126.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="142.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">F</text>
<text x="142.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="166.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="182.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="182.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="200.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="216.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="22.0" y="256.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="30.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="57.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<text x="57.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="90.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="117.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<text x="117.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="150.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="177.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H2</text>
<text x="177.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="210.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="237.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<text x="237.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="270.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="297.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="297.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="330.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="357.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="357.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="34.0" y1="218.0" x2="57.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="68.0" y1="218.0" x2="117.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="102.0" y1="218.0" x2="177.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="142.0" y1="218.0" x2="237.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="182.0" y1="218.0" x2="297.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="216.0" y1="218.0" x2="357.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="22.0" y="328.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="22.0" y="344.0" font-size="9.5" fill="#1c2833">A：H0–H2 的 free = 0，在用</text>
<text x="22.0" y="359.0" font-size="9.5" fill="#1c2833">F（H3）、C（H4–H5）：free = 1，映射还在</text>
<text x="22.0" y="374.0" font-size="9.5" fill="#1c2833">reserved = 12 MB（6 个 chunk）</text>
<rect x="422.0" y="92.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="434.0" y="114.0" font-size="13" font-weight="bold" fill="#1c2833">② 释放 A：H0–H2 变空闲</text>
<text x="434.0" y="130.0" font-size="10" fill="#33414e">A 的 Block 变空闲，H0–H2 的 free 置 1。映射没有动。</text>
<text x="434.0" y="176.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="430.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="446.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="446.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="464.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="480.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="498.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="514.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="538.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="554.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">F</text>
<text x="554.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="578.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="594.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="594.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="612.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="628.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="434.0" y="256.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="442.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="469.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="469.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="502.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="529.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="529.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="562.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="589.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="589.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="622.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="649.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<text x="649.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="682.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="709.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="709.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="742.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="769.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="769.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="446.0" y1="218.0" x2="469.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="480.0" y1="218.0" x2="529.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="514.0" y1="218.0" x2="589.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="554.0" y1="218.0" x2="649.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="594.0" y1="218.0" x2="709.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="628.0" y1="218.0" x2="769.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="434.0" y="328.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="434.0" y="344.0" font-size="9.5" fill="#1c2833">Block（T_A）.allocated：1→0</text>
<text x="434.0" y="359.0" font-size="9.5" fill="#1c2833">H0–H2：free 0→1；映射仍是 1 条</text>
<text x="434.0" y="374.0" font-size="9.5" fill="#1c2833">现在 H0–H5 都空闲，但 A、F、C 是三段互不相连的 VA</text>
<rect x="834.0" y="92.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="846.0" y="114.0" font-size="13" font-weight="bold" fill="#1c2833">③ H3 被分配；请求 T4 = 8 MB</text>
<text x="846.0" y="130.0" font-size="10" fill="#33414e">F 分给新张量 B（H3 在用）。T4 要连续 4 块：空闲里最大连续只有 3 块（A），</text>
<text x="846.0" y="144.0" font-size="10" fill="#33414e">但空闲总共 5 块（A + C）。</text>
<text x="846.0" y="176.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="842.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="858.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="858.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="876.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="892.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="910.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="926.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="950.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="966.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">B</text>
<text x="966.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="990.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1006.0" y="197.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="1006.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="1024.0" y="184.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1040.0" y="213.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="846.0" y="256.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="854.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="881.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="881.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="914.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="941.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="941.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="974.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1001.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="1001.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="1034.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="1061.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="1061.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="1094.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1121.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="1121.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="1154.0" y="262.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1181.0" y="281.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="1181.0" y="304.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="858.0" y1="218.0" x2="881.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="892.0" y1="218.0" x2="941.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="926.0" y1="218.0" x2="1001.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="966.0" y1="218.0" x2="1061.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1006.0" y1="218.0" x2="1121.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1040.0" y1="218.0" x2="1181.0" y2="260.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="846.0" y="328.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="846.0" y="344.0" font-size="9.5" fill="#1c2833">B：H3 的 free = 0，在用</text>
<text x="846.0" y="359.0" font-size="9.5" fill="#1c2833">连续空闲最多 3 块（A），不够 4 块</text>
<text x="846.0" y="374.0" font-size="9.5" fill="#1c2833">空闲总数 5 块 ≥ 4 块：可以不新分配。分叉点。</text>
<rect x="10.0" y="446.0" width="1220.0" height="22.0" rx="4" fill="#1e8449" stroke="none" stroke-width="1"/>
<text x="20.0" y="461.0" font-size="11" font-weight="bold" fill="#ffffff">VMS 支线：④ 拼接 → ⑤ T4 释放 → ⑥ GC 解除 V4 的映射</text>
<rect x="10.0" y="474.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="22.0" y="496.0" font-size="13" font-weight="bold" fill="#1c2833">④ 拼接：A + C 的 H4，新建 V4（4 块）</text>
<text x="22.0" y="512.0" font-size="10" fill="#33414e">取 A（3 块）和 C 的 H4 凑出 4 块；C 多出的 H5 切开放回空闲。V4 是连续 8 </text>
<text x="22.0" y="526.0" font-size="10" fill="#33414e">MB 的 VA，没有新的 HBM。</text>
<text x="22.0" y="558.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="18.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3" stroke-dasharray="4 3"/>
<text x="34.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="34.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="52.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3" stroke-dasharray="4 3"/>
<text x="68.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="86.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3" stroke-dasharray="4 3"/>
<text x="102.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="126.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="142.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">B</text>
<text x="142.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="166.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3" stroke-dasharray="4 3"/>
<text x="182.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="182.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="200.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3" stroke-dasharray="4 3"/>
<text x="216.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<rect x="240.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="256.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">V4</text>
<text x="256.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="274.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="290.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="308.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="324.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="342.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="358.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="22.0" y="638.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="30.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="57.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<text x="57.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m2</text>
<rect x="90.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="117.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<text x="117.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m2</text>
<rect x="150.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="177.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H2</text>
<text x="177.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m2</text>
<rect x="210.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="237.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="237.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="270.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="297.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H4</text>
<text x="297.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m2</text>
<rect x="330.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="357.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="357.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="34.0" y1="600.0" x2="57.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="256.0" y1="600.0" x2="57.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="68.0" y1="600.0" x2="117.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="290.0" y1="600.0" x2="117.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="102.0" y1="600.0" x2="177.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="324.0" y1="600.0" x2="177.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="142.0" y1="600.0" x2="237.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="182.0" y1="600.0" x2="297.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="358.0" y1="600.0" x2="297.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="216.0" y1="600.0" x2="357.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="22.0" y="710.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="22.0" y="726.0" font-size="9.5" fill="#1c2833">V4.mapped：0→1；V4.fused = 1</text>
<text x="22.0" y="741.0" font-size="9.5" fill="#1c2833">H0–H2、H4：free 1→0；mapped_blocks 1→2（多了 V4）</text>
<text x="22.0" y="756.0" font-size="9.5" fill="#1c2833">H5：切开后仍空闲，m = 1</text>
<text x="22.0" y="771.0" font-size="9.5" fill="#1c2833">A、C 的 Block 被 V4 占用，映射未撤销</text>
<rect x="422.0" y="474.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="434.0" y="496.0" font-size="13" font-weight="bold" fill="#1c2833">⑤ T4 释放</text>
<text x="434.0" y="512.0" font-size="10" fill="#33414e">V4 的 Block 变空闲，H0–H2、H4 标 free = 1；A、C 的 Block 随之空闲。映</text>
<text x="434.0" y="526.0" font-size="10" fill="#33414e">射都还在。</text>
<text x="434.0" y="558.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="430.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="446.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="446.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="464.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="480.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="498.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="514.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="538.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="554.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">B</text>
<text x="554.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="578.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="594.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="594.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="612.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="628.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<rect x="652.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="668.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">V4</text>
<text x="668.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="686.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="702.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="720.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="736.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="754.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="770.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="434.0" y="638.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="442.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="469.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="469.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m2</text>
<rect x="502.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="529.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="529.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m2</text>
<rect x="562.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="589.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="589.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m2</text>
<rect x="622.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="649.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="649.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="682.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#ca6f1e" stroke-width="2.4"/>
<text x="709.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="709.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m2</text>
<rect x="742.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="769.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="769.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="446.0" y1="600.0" x2="469.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="668.0" y1="600.0" x2="469.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="480.0" y1="600.0" x2="529.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="702.0" y1="600.0" x2="529.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="514.0" y1="600.0" x2="589.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="736.0" y1="600.0" x2="589.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="554.0" y1="600.0" x2="649.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="594.0" y1="600.0" x2="709.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="770.0" y1="600.0" x2="709.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="628.0" y1="600.0" x2="769.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="434.0" y="710.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="434.0" y="726.0" font-size="9.5" fill="#1c2833">V4.allocated：1→0；A、C.allocated：1→0</text>
<text x="434.0" y="741.0" font-size="9.5" fill="#1c2833">H0–H2、H4：free 0→1；m 仍是 2</text>
<text x="434.0" y="756.0" font-size="9.5" fill="#1c2833">V4、A、C 的 mapped 仍是 1：还没有 unmap</text>
<rect x="834.0" y="474.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="846.0" y="496.0" font-size="13" font-weight="bold" fill="#1c2833">⑥ GC：解除 V4 的映射</text>
<text x="846.0" y="512.0" font-size="10" fill="#33414e">触发 GC（emptyCache、autoGC 超限，或分配失败后的后续重试）。V4 的 event</text>
<text x="846.0" y="526.0" font-size="10" fill="#33414e"> 完成，它的映射被撤销。</text>
<text x="846.0" y="558.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="842.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="858.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">A</text>
<text x="858.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="876.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="892.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="910.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="926.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="950.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="966.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">B</text>
<text x="966.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="990.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1006.0" y="579.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">C</text>
<text x="1006.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="1024.0" y="566.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1040.0" y="595.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="846.0" y="638.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="854.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="881.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="881.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="914.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="941.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="941.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="974.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1001.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="1001.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="1034.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="1061.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="1061.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f0 m1</text>
<rect x="1094.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1121.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="1121.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<rect x="1154.0" y="644.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1181.0" y="663.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="1181.0" y="686.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">f1 m1</text>
<line x1="858.0" y1="600.0" x2="881.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="892.0" y1="600.0" x2="941.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="926.0" y1="600.0" x2="1001.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="966.0" y1="600.0" x2="1061.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1006.0" y1="600.0" x2="1121.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1040.0" y1="600.0" x2="1181.0" y2="642.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="846.0" y="710.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="846.0" y="726.0" font-size="9.5" fill="#1c2833">V4.released：0→1（cuMemUnmap + 释放 VA）</text>
<text x="846.0" y="741.0" font-size="9.5" fill="#1c2833">H0–H2、H4：m 2→1（删掉 V4 的记录）</text>
<text x="846.0" y="756.0" font-size="9.5" fill="#1c2833">A、C 仍映射它们（它们自己的映射没有撤销）</text>
<rect x="10.0" y="828.0" width="1220.0" height="22.0" rx="4" fill="#ca6f1e" stroke="none" stroke-width="1"/>
<text x="20.0" y="843.0" font-size="11" font-weight="bold" fill="#ffffff">页回收支线：④ unmap 空洞的页 → ⑤ map 到段尾 → ⑥ T4 释放</text>
<rect x="10.0" y="856.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="22.0" y="878.0" font-size="13" font-weight="bold" fill="#1c2833">④ unmap 空洞里的 idle 页 H0、H1、H2、H4</text>
<text x="22.0" y="894.0" font-size="10" fill="#33414e">recycler 从 5 个空闲页里挑 4 个（H0、H1、H2、H4，8 MB），从空洞上 unmap</text>
<text x="22.0" y="908.0" font-size="10" fill="#33414e">。空洞 VA 变空，不占 HBM。H5 仍空闲，映射在 S5。</text>
<text x="22.0" y="940.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="18.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="34.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S0</text>
<text x="34.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="52.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="68.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S1</text>
<text x="68.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="86.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="102.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S2</text>
<text x="102.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="120.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="136.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S3</text>
<text x="136.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="154.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="170.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S4</text>
<text x="170.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<rect x="188.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="204.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S5</text>
<text x="204.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<rect x="222.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="238.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S6</text>
<text x="238.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="256.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="272.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S7</text>
<text x="272.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="290.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="306.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S8</text>
<text x="306.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="324.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="340.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S9</text>
<text x="340.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<text x="22.0" y="1020.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="30.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="57.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="57.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s—</text>
<rect x="90.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="117.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="117.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s—</text>
<rect x="150.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="177.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="177.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s—</text>
<rect x="210.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="237.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="237.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s3</text>
<rect x="270.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="297.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="297.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s—</text>
<rect x="330.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="357.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="357.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s5</text>
<line x1="34.0" y1="982.0" x2="57.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="68.0" y1="982.0" x2="117.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="102.0" y1="982.0" x2="177.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="136.0" y1="982.0" x2="237.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="170.0" y1="982.0" x2="297.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="204.0" y1="982.0" x2="357.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="22.0" y="1092.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="22.0" y="1108.0" font-size="9.5" fill="#1c2833">slot.mapped：S0–S2、S4 各 1→0（unmap）</text>
<text x="22.0" y="1123.0" font-size="9.5" fill="#1c2833">page.owner：H0、H1、H2、H4 → —（未映射）</text>
<text x="22.0" y="1138.0" font-size="9.5" fill="#1c2833">reserved 仍 12 MB（没有 cudaFree）</text>
<rect x="422.0" y="856.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="434.0" y="878.0" font-size="13" font-weight="bold" fill="#1c2833">⑤ map 到段尾 S6–S9</text>
<text x="434.0" y="894.0" font-size="10" fill="#33414e">这 4 个页映射到段尾，形成连续 8 MB（S6–S9）给 T4 用。没有 cudaMalloc。</text>
<text x="434.0" y="940.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="430.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="446.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S0</text>
<text x="446.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="464.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="480.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S1</text>
<text x="480.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="498.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="514.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S2</text>
<text x="514.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="532.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="548.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S3</text>
<text x="548.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="566.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="582.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S4</text>
<text x="582.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="600.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="616.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S5</text>
<text x="616.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<rect x="634.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="650.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S6</text>
<text x="650.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="668.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="684.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S7</text>
<text x="684.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="702.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="718.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S8</text>
<text x="718.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="736.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="752.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S9</text>
<text x="752.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="434.0" y="1020.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="442.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="469.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<text x="469.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s6</text>
<rect x="502.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="529.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<text x="529.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s7</text>
<rect x="562.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="589.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H2</text>
<text x="589.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s8</text>
<rect x="622.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="649.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="649.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s3</text>
<rect x="682.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="709.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H4</text>
<text x="709.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s9</text>
<rect x="742.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="769.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="769.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s5</text>
<line x1="548.0" y1="982.0" x2="649.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="616.0" y1="982.0" x2="769.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="650.0" y1="982.0" x2="469.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="684.0" y1="982.0" x2="529.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="718.0" y1="982.0" x2="589.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="752.0" y1="982.0" x2="709.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="434.0" y="1092.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="434.0" y="1108.0" font-size="9.5" fill="#1c2833">slot.mapped：S6–S9 各 0→1（map）</text>
<text x="434.0" y="1123.0" font-size="9.5" fill="#1c2833">page.owner：H0→S6，H1→S7，H2→S8，H4→S9</text>
<text x="434.0" y="1138.0" font-size="9.5" fill="#1c2833">page.idle：H0、H1、H2、H4 各 1→0；reserved 仍 12 MB</text>
<rect x="834.0" y="856.0" width="400.0" height="340.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1"/>
<text x="846.0" y="878.0" font-size="13" font-weight="bold" fill="#1c2833">⑥ T4 释放</text>
<text x="846.0" y="894.0" font-size="10" fill="#33414e">T4 的页 fence 完成，idle = 1。页仍映射在 S6–S9，没有 unmap。H5 也还空闲</text>
<text x="846.0" y="908.0" font-size="10" fill="#33414e">。</text>
<text x="846.0" y="940.0" font-size="9" font-weight="bold" fill="#3d2466">VA（VMS：VirDevPtr 块；页回收：段内的格子，每格 2 MB）</text>
<rect x="842.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="858.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S0</text>
<text x="858.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="876.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="892.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S1</text>
<text x="892.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="910.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="926.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S2</text>
<text x="926.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="944.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
<text x="960.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S3</text>
<text x="960.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<rect x="978.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#ffffff" stroke="#aab7b8" stroke-width="1.3" stroke-dasharray="3 3"/>
<text x="994.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S4</text>
<text x="994.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">—</text>
<rect x="1012.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1028.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S5</text>
<text x="1028.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<rect x="1046.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1062.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S6</text>
<text x="1062.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="1080.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1096.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S7</text>
<text x="1096.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<rect x="1114.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1130.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S8</text>
<text x="1130.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="1148.0" y="948.0" width="32.0" height="34.0" rx="3" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.3"/>
<text x="1164.0" y="961.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">S9</text>
<text x="1164.0" y="977.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="846.0" y="1020.0" font-size="9" font-weight="bold" fill="#333333">HBM（物理 chunk，每格 2 MB；例子只用到 H0–H5）</text>
<rect x="854.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="881.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<text x="881.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s6</text>
<rect x="914.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="941.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H1</text>
<text x="941.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s7</text>
<rect x="974.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1001.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<text x="1001.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s8</text>
<rect x="1034.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#7d3c98" stroke="#4d4d4d" stroke-width="1"/>
<text x="1061.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H3</text>
<text x="1061.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i0 s3</text>
<rect x="1094.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1121.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H4</text>
<text x="1121.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s9</text>
<rect x="1154.0" y="1026.0" width="54.0" height="30.0" rx="3" fill="#d5d8dc" stroke="#4d4d4d" stroke-width="1"/>
<text x="1181.0" y="1045.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H5</text>
<text x="1181.0" y="1068.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#33414e">i1 s5</text>
<line x1="960.0" y1="982.0" x2="1061.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1028.0" y1="982.0" x2="1181.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1062.0" y1="982.0" x2="881.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1096.0" y1="982.0" x2="941.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1130.0" y1="982.0" x2="1001.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="1164.0" y1="982.0" x2="1121.0" y2="1024.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="846.0" y="1092.0" font-size="9" font-weight="bold" fill="#3d2466">变化的 flag</text>
<text x="846.0" y="1108.0" font-size="9.5" fill="#1c2833">page.idle：H0、H1、H2、H4 各 0→1</text>
<text x="846.0" y="1123.0" font-size="9.5" fill="#1c2833">slot.mapped 仍是 1（没有 unmap）</text>
<text x="846.0" y="1138.0" font-size="9.5" fill="#1c2833">下一次 miss 时才会再搬（这里不展开）</text>
<rect x="10.0" y="1212.0" width="1220.0" height="22.0" rx="4" fill="#3d2466" stroke="none" stroke-width="1"/>
<text x="20.0" y="1227.0" font-size="11" font-weight="bold" fill="#ffffff">区别（④ 之后）</text>
<rect x="10.0" y="1238.0" width="1220.0" height="22.0" rx="3" fill="#f2f4f4" stroke="none" stroke-width="1"/>
<text x="18.0" y="1253.0" font-size="11" font-weight="bold" fill="#1c2833">项目</text>
<text x="188.0" y="1253.0" font-size="11" font-weight="bold" fill="#1c2833">VMS</text>
<text x="718.0" y="1253.0" font-size="11" font-weight="bold" fill="#1c2833">页回收</text>
<text x="18.0" y="1276.0" font-size="10.5" font-weight="bold" fill="#1c2833">第 ④ 步做什么</text>
<text x="188.0" y="1276.0" font-size="10.5" fill="#1c2833">VMS：拼接 A（H0–H2）和 C 的 H4，新建 V4（8 MB），映射 H0、H1、H2、H4；H5 切开放回空闲。没有新</text>
<text x="188.0" y="1290.0" font-size="10.5" fill="#1c2833">的 HBM。</text>
<text x="718.0" y="1276.0" font-size="10.5" fill="#1c2833">页回收：unmap S0–S2、S4 的页（H0、H1、H2、H4），再 map 到段尾 S6–S9。没有新的 HBM。</text>
<rect x="10.0" y="1298.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1314.0" font-size="10.5" font-weight="bold" fill="#1c2833">T4 的 VA 在哪</text>
<text x="188.0" y="1314.0" font-size="10.5" fill="#1c2833">VMS：一段新的 VA（V4），地址不要求在段尾。</text>
<text x="718.0" y="1314.0" font-size="10.5" fill="#1c2833">页回收：段尾 S6–S9，是段内原有的地址。</text>
<rect x="10.0" y="1322.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1338.0" font-size="10.5" font-weight="bold" fill="#1c2833">空洞的 VA 之后</text>
<text x="188.0" y="1338.0" font-size="10.5" fill="#1c2833">VMS：仍映射着 H0–H2、H4（A、C 的映射保留）。</text>
<text x="718.0" y="1338.0" font-size="10.5" fill="#1c2833">页回收：变成未映射的空 VA，没有物理页。</text>
<rect x="10.0" y="1346.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1362.0" font-size="10.5" font-weight="bold" fill="#1c2833">每块 HBM 的映射数</text>
<text x="188.0" y="1362.0" font-size="10.5" fill="#1c2833">VMS：H0–H2、H4 各 2 条（同一个 chunk 被两段 VA 映射）。</text>
<text x="718.0" y="1362.0" font-size="10.5" fill="#1c2833">页回收：每块 1 条。</text>
<rect x="10.0" y="1370.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1386.0" font-size="10.5" font-weight="bold" fill="#1c2833">物理页的 VA</text>
<text x="188.0" y="1386.0" font-size="10.5" fill="#1c2833">VMS：页不动，只是多了一条映射。</text>
<text x="718.0" y="1386.0" font-size="10.5" fill="#1c2833">页回收：页被搬到了新的 VA（H0 从 S0 到 S6，H1 从 S1 到 S7，依此类推）。</text>
<rect x="10.0" y="1394.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1410.0" font-size="10.5" font-weight="bold" fill="#1c2833">什么时候 unmap</text>
<text x="188.0" y="1410.0" font-size="10.5" fill="#1c2833">VMS：⑥（GC）。T4 释放之后，event 完成才做。</text>
<text x="718.0" y="1410.0" font-size="10.5" fill="#1c2833">页回收：④（miss 时）。搬页时 unmap 空洞的页。</text>
<rect x="10.0" y="1418.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1434.0" font-size="10.5" font-weight="bold" fill="#1c2833">T4 释放之后</text>
<text x="188.0" y="1434.0" font-size="10.5" fill="#1c2833">VMS：映射都还在；GC 时才撤销 V4 的映射。</text>
<text x="718.0" y="1434.0" font-size="10.5" fill="#1c2833">页回收：页空闲，仍映射在 S6–S9；H5 空闲，仍在 S5。下一次 miss 才会再搬。</text>
<rect x="10.0" y="1442.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="18.0" y="1458.0" font-size="10.5" font-weight="bold" fill="#1c2833">reserved</text>
<text x="188.0" y="1458.0" font-size="10.5" fill="#1c2833">VMS 和页回收 都是 12 MB，没有 cudaMalloc。</text>
<text x="718.0" y="1458.0" font-size="10.5" fill="#1c2833">（不拼接 / 不回收时要新建 4 个 chunk，reserved 从 12 MB 涨到 20 MB。）</text>
<rect x="10.0" y="1466.0" width="1220.0" height="0.6" rx="0" fill="#ccd1d1" stroke="none" stroke-width="1"/>
<text x="10.0" y="1490.0" font-size="10" fill="#566573">假设：VMS 的 fragLimit = 0 演示拼接（默认 512 MB 时 8 MB 不拼接，会新建 4 个 chunk）；页回收的 reserved cap = 10 MB（reserved 12 MB 超过它，回收开启）。</text>
<text x="10.0" y="1505.0" font-size="10" fill="#566573">页 = 2 MB；地址示意；手推的，未实际运行。两边的 ④、⑤、⑥ 不是一一对应的步骤。</text>
</svg>
```
