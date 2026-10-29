# torch.cuda.empty_cache() 的具体实现：调用链、两遍释放与例子（emptycache_layers_ops）

新 session 只需读本 paste。

- 基于：无（首版）
- 对应文件：`emptycache_layers_ops.svg`（源文件，内容同最后一节）

## 核心论点

**一句话目的：** 说清楚 `torch.cuda.empty_cache()` 在 PyTorch 原生分配器里走哪些函数、分哪两遍释放、每一步调哪些驱动调用，以及释放之后碎片还剩什么。

**核心知识点（草稿，待你确认）：** emptyCache 只释放池里的空闲块。`release_blocks` 分两遍：第一遍整块普通块直接 `cudaFree`，expandable 块先收进 `to_unmap`；第二遍逐个 unmap、release，段空了再整段删除（`cuMemAddressFree`）。在用的块不在池里，不会被释放，所以释放之后段还在，碎片也还在。

**整张图的精髓：** 上半是调用链（1–10 的顺序，4a、4b 对应到每一步），下半是例子的状态链：每个框是一个状态，框之间的数字是调用，红色虚线是两遍的分界。

**措辞说明：** 例子的状态是手推的（未运行）。cudaFree 内部调用的 cuMem\* 图上没有连线，原因是 runtime 闭源，只能 trace 才能确定。small 池、graph pool 分支、cudaDeviceSynchronize 分支没有画。

## 术语

| 术语 | 含义 |
|---|---|
| 池（pool） | DeviceCachingAllocator 里分 large、small 两个 BlockPool，只放空闲块 |
| to_unmap | release_blocks 第 1 遍收集的 expandable 空闲块，第 2 遍逐个取出处理 |
| 段 E | 一个 ExpandableSegment，整段 VA 构造时预留，槽位（V0、V1、V2…）按需映射 |
| V 槽位 | 段 E 里的一段 VA，由 PyTorch 映射、解除、释放 |
| V3 | 普通块的 VA（cudaMalloc 返回的指针），由驱动分配和释放 |
| H 块 | 一块 HBM（2 MB 物理句柄），在用 / 空闲 / 已归还 |
| reserved | 分配器持有的物理内存；只在 cudaFree 和 cuMemRelease 时下降 |

## 知识

| ID | 内容 | 来源 | 状态 | 确认 | 你的理解 | 对应图元 |
|---|---|---|---|---|---|---|
| K1 | emptyCache 调用链：torch.cuda.empty_cache()（memory.py:234）→ THCPModule_emptyCache（Module.cpp:575）→ CUDACachingAllocator::emptyCache（h:395，内联）→ NativeCachingAllocator::emptyCache（:5312，逐设备）→ DeviceCachingAllocator::emptyCache（:2763）→ release_cached_blocks（:4524）。 | fbsource torch/cuda/memory.py、torch/csrc/cuda/Module.cpp、c10/cuda/CUDACachingAllocator.{h,cpp} | verified（读过代码） | 待确认 |  | 第 1–5 层的方框 |
| K2 | release_cached_blocks：只有默认 pool 且非 graph 捕获时，才先调 synchronize_and_free_events（每个待完成的 event 一次 cudaEventSynchronize，再 free_block）。然后对 large、small 两个池各调一次 release_blocks，再处理 graph_pools_freeable。 | CUDACachingAllocator.cpp :4524–4560；synchronize_and_free_events 约 :4750–4790 | verified（读过代码） | 待确认 |  | 6 号方框；7 的 large 先于 small |
| K3 | release_blocks 分两遍。第 1 遍（7a）遍历 pool.blocks：expandable 块只 push 进 to_unmap；整块普通块（无 prev、next）直接 release_block（8）。第 2 遍（7b）逐个处理 to_unmap：unmap_block（9）；段因此变空才 release_expandable_segment（10）。遍历时不能 unmap，因为 unmap 会改动 free pool（代码注释）。 | CUDACachingAllocator.cpp :4718–4741 | verified（读过代码） | 待确认 |  | 7a、7b 两遍及 to_unmap 框 |
| K4 | pool 里只有空闲块：free_block 插入池（:3992），分配时 get_free_block 从池里取出（:4214）。所以 emptyCache 不会碰在用的块。 | CUDACachingAllocator.cpp :3992、:4214 | verified（读过代码） | 待确认 |  | 例子里在用的 H1 不动 |
| K5 | release_block（:4584）：非 expandable 块，调 cudaFree，并减 reserved 统计。 | CUDACachingAllocator.cpp :4584–4612 | verified（读过代码） | 待确认 |  | 8 号方框（cudaFree） |
| K6 | unmap_block（:4640）→ ExpandableSegment::unmap → unmapHandles（:999）。unmapHandles 先同步（有 stream 时 cudaStreamSynchronize，否则 cudaDeviceSynchronize），然后对每个 handle 依次 cuMemUnmap（:1022）、cuMemRelease（:1038）。 | CUDACachingAllocator.cpp :999–1040、:4640–4660 | verified（读过代码） | 待确认 |  | 9、9a、9b、9c 方框 |
| K7 | release_expandable_segment（:4570）：从 expandable_segments_ 移除并 delete 段对象；析构 ~ExpandableSegment 里 cuMemAddressFree（:929）。段的 VA 是构造时用 cuMemAddressReserve（:536）整段预留的。 | CUDACachingAllocator.cpp :536、:570–580、:929 | verified（读过代码） | 待确认 |  | 10 号方框（cuMemAddressFree） |
| K8 | cudaFree 内部调用了哪些 cuMem*：runtime 闭源，PyTorch 源码里看不到，无法联网查文档。需要在 GPU 上 trace（CUPTI driver callback 或 nsys）才能确定。 | 未查到（无外网，无 GPU） | unverified | 待确认 |  | 4a 的 cudaFree 没有连到 4b（只有虚线到总线） |
| K9 | cuMemUnmap 只删映射，不释放 HBM；cuMemRelease 释放物理句柄，HBM 归还。代码里的顺序（先 unmap 后 release）是 verified；驱动层语义是通用知识。 | 代码：unmapHandles :1022、:1038；语义：CUDA 驱动（文档未核对） | verified（顺序）／unverified（语义） | 待确认 |  | 9b、9c 方框；S2–S5 的 HBM 变化 |
| K10 | 普通块的 VA（cudaMalloc 返回的指针）由驱动分配和释放，PyTorch 不记账。cudaFree 后这块 VA 消失。 | PyTorch 源码里没有对应的 VA 操作；驱动行为未核对 | unverified（通用，PyTorch 侧 verified 无记账） | 待确认 |  | S0–S1 的 V3 |
| K11 | 段 E 的整段 VA 由 PyTorch 管：构造时预留，槽位按需映射和解除，整段在段删除时释放。这就是图上"段 E（PyTorch 管）"的意思。 | K7 + CUDACachingAllocator.cpp 构造 :536、映射 :963–983、解除 :1022 | verified（读过代码） | 待确认 |  | S0–S5 的 V0–V2 标签 |
| K12 | 例子（手推，未运行）：起始段 E 有 V0→H0（空闲）、V1→H1（在用）、V2→H2（空闲），普通块 H3（空闲）。第 1 遍 cudaFree(H3)，并 push V0、V2。第 2 遍 unmap V0、release H0、unmap V2、release H2。H1 在用，段不空，所以不 cuMemAddressFree。 | 按 K3、K4、K6、K7 手推 | inferred（手推） | 待确认 |  | S0–S5 状态链 |
| K13 | 例子结果：reserved 从 8 MB 降到 2 MB（只剩在用的 H1）；V0、V2 变成不相邻的未映射空洞，4 MB 的连续 VA 请求用不了，只能在段尾新映射。碎片没有消失，只是 HBM 还了回去。 | 按 K12 推出；expandable 段语义 | inferred（手推） | 待确认 |  | S5 结束 与底部说明 |
| K14 | 文档（Page recycling 6b，Jasper Shan）：页回收在 miss 时搬页，不调用 empty_cache，也不释放段。文档里 empty_cache 是单独做的一次性实验（生产上 reserved 降到 58.7 GiB，96 秒内回涨，最后比原平台高 4 GiB），在修复表里标为 measured negative。 | meta search.knowledge load（Google Doc，6b 节、一次性 empty_cache 实验、修复表） | verified（读过文档） | 待确认 |  | 与 page recycling 的区别 |
| K15 | GMLake patch 的 emptyCache 在 release_cached_blocks 之后还会调 garbage_collect_fused_blocks(2, 0)（:1751–1755）。PyTorch 原生的 emptyCache 没有这一步。 | GMLake CUDACachingAllocator.cpp :1751–1755（patch 版） | verified（读过代码） | 待确认 |  | （图里已删除，只留在本条） |
| K16 | 第 5 层（页表删除、物理页归还、MMU）是通用知识，未核对。 | 通用 | unverified | 待确认 |  | 第 5 层方框 |

状态：verified（读过代码或文档）／inferred（手推的例子，未运行）／unverified（通用机制，或文档原文未对照代码）。确认：待确认 / 已确认。

## 分组讲解（待确认）

### 第 1 组：谁进入释放路径（K1–K4）

- 调用链 K1，然后 release_cached_blocks（K2）。
- 池里只有空闲块（K4），所以在用的块不会被碰。

检查题：如果一个块还有没完成的 event，emptyCache 会怎么处理？

### 第 2 组：两遍释放（K3、K5–K7、K9）

- 第 1 遍：普通整块 → cudaFree（K5）；expandable 块 → to_unmap。
- 第 2 遍：逐个 unmap、release（K6、K9）；段空了才 cuMemAddressFree（K7）。
- 两遍之间的顺序是代码结构决定的，遍历时不能 unmap。

检查题：如果段 E 的 H1 在 emptyCache 时还在用，段为什么不能整段释放？（K3、K7、K13）

### 第 3 组：例子和碎片（K10–K13）

- 起始：段 E 有一块在用块夹在两段空闲之间（碎片）。
- 结束：reserved 2 MB，V0、V2 是不相邻的空洞，连续 4 MB 请求用不了（K13）。
- 与 doc 的 page recycling 的区别：页回收搬页，emptyCache 只还 HBM（K14）。

## 图的内容

1. 上半：分层调用链（1–5 层），角标 1–10 是调用顺序，7a 与 7b 是两遍，to_unmap 是两遍之间的存储框。
2. 下半：例子的 6 个状态（起始、cudaFree H3 后、三步 unmap/release 后、结束），每个框有 VA、HBM、reserved、to_unmap。两遍之间有红色虚线。
3. 框外的说明只留三行：cudaFree、cuMemRelease、cuMemAddressFree、cuMemUnmap 各自释放什么，以及"只处理池里的空闲块"和碎片结果。

## 结构

- 布局：上半是分层调用图（1–5 层，4a、4b 在下面），下半是例子的状态行。
- 颜色：紫色 = 在用的 HBM；灰色 = 空闲（仍持有）；虚线空框 = 已归还；绿色 = 已映射；空心 = 预留未映射；黄色 = to_unmap。

## SVG 约定

- 尺寸：`viewBox="0 0 1240 1162"`
- 字体：`'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial`
- 方框编号 1–10 是调用顺序；V、H 编号是示意。

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="1162" viewBox="0 0 1240 1162" font-family="'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#1c2833"/></marker></defs>
<rect width="1240" height="1162" fill="#ffffff"/>
<text x="10.0" y="28.0" font-size="16" font-weight="bold" fill="#1c2833">torch.cuda.empty_cache() 的具体实现</text>
<text x="10.0" y="50.0" font-size="10.5" fill="#566573">Python → C++ 绑定 → 分配器 → release_blocks 两遍 → 驱动 / runtime。编号 = 调用顺序；实线 = 显式调用，虚线 = 隐式。</text>
<rect x="8.0" y="56.0" width="1224.0" height="62.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="14.0" y="69.0" font-size="12" font-weight="bold" fill="#34495e">1  用户 / Python / C++ 绑定</text>
<rect x="8.0" y="128.0" width="1224.0" height="342.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="14.0" y="141.0" font-size="12" font-weight="bold" fill="#34495e">2  PyTorch 分配器</text>
<rect x="8.0" y="482.0" width="1224.0" height="54.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="14.0" y="495.0" font-size="12" font-weight="bold" fill="#34495e">3  驱动封装</text>
<rect x="8.0" y="550.0" width="1224.0" height="60.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="14.0" y="563.0" font-size="12" font-weight="bold" fill="#34495e">4a  CUDA Runtime</text>
<rect x="8.0" y="628.0" width="1224.0" height="92.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="14.0" y="641.0" font-size="12" font-weight="bold" fill="#34495e">4b  CUDA Driver API（libcuda.so）</text>
<rect x="8.0" y="732.0" width="1224.0" height="58.0" rx="4" fill="#fbfcfc" stroke="#e5e8e8" stroke-width="0.8"/>
<text x="1218.0" y="745.0" font-size="12" font-weight="bold" text-anchor="end" fill="#34495e">5  内核驱动 + GPU 硬件</text>
<rect x="45.0" y="74.0" width="210.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="150.0" y="92.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">torch.cuda.empty_cache()</text>
<rect x="300.0" y="74.0" width="200.0" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="400.0" y="92.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">THCPModule_emptyCache</text>
<rect x="525.4" y="74.0" width="229.2" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="640.0" y="92.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">CUDACachingAllocator::emptyCache</text>
<rect x="518.8" y="150.0" width="242.4" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="640.0" y="168.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">DeviceCachingAllocator::emptyCache</text>
<rect x="560.0" y="206.0" width="160.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="640.0" y="224.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">release_cached_blocks</text>
<rect x="65.0" y="206.0" width="190.0" height="28.0" rx="5" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.2"/>
<text x="160.0" y="224.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">synchronize_and_free_events</text>
<rect x="540.0" y="262.0" width="200.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="640.0" y="280.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">release_blocks · 第1遍</text>
<rect x="540.0" y="318.0" width="200.0" height="28.0" rx="5" fill="#fef9e7" stroke="#b7950b" stroke-width="1.2"/>
<text x="640.0" y="336.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">to_unmap（expandable 块）</text>
<rect x="970.0" y="262.0" width="140.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="1040.0" y="280.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">release_block</text>
<rect x="540.0" y="374.0" width="200.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="640.0" y="392.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">release_blocks · 第2遍</text>
<rect x="335.0" y="374.0" width="130.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="400.0" y="392.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">unmap_block</text>
<rect x="290.0" y="430.0" width="220.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="400.0" y="448.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">unmapHandles()</text>
<rect x="552.0" y="430.0" width="176.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="640.0" y="448.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">release_expandable_segment</text>
<rect x="440.0" y="502.0" width="560.0" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="720.0" y="520.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">DriverAPI（cuMem*_）</text>
<rect x="1005.0" y="576.0" width="70.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="1040.0" y="594.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cudaFree</text>
<rect x="241.7" y="576.0" width="156.6" height="28.0" rx="5" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.2"/>
<text x="320.0" y="594.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cudaStreamSynchronize</text>
<rect x="75.0" y="576.0" width="150.0" height="28.0" rx="5" fill="#eaecee" stroke="#7f8c8d" stroke-width="1.2"/>
<text x="150.0" y="594.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cudaEventSynchronize</text>
<rect x="40.0" y="684.0" width="84.0" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="82.0" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemUnmap</text>
<rect x="197.3" y="684.0" width="97.2" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="245.9" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemRelease</text>
<rect x="367.9" y="684.0" width="123.6" height="28.0" rx="5" fill="#fadbd8" stroke="#c0392b" stroke-width="1.2"/>
<text x="429.7" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemAddressFree</text>
<rect x="564.8" y="684.0" width="90.6" height="28.0" rx="5" fill="#ffffff" stroke="#aab7b8" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="610.1" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">cuMemCreate</text>
<rect x="728.7" y="684.0" width="143.4" height="28.0" rx="5" fill="#ffffff" stroke="#aab7b8" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="800.4" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">cuMemAddressReserve</text>
<rect x="945.5" y="684.0" width="70.8" height="28.0" rx="5" fill="#ffffff" stroke="#aab7b8" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="980.9" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">cuMemMap</text>
<rect x="1089.6" y="684.0" width="110.4" height="28.0" rx="5" fill="#ffffff" stroke="#aab7b8" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="1144.8" y="702.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">cuMemSetAccess</text>
<rect x="12.0" y="750.0" width="140.0" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="82.0" y="768.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">页表项删除（隐式）</text>
<rect x="160.9" y="750.0" width="170.0" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="245.9" y="768.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">物理页归还（隐式）</text>
<rect x="344.7" y="750.0" width="170.0" height="28.0" rx="5" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1.2"/>
<text x="429.7" y="768.0" font-size="11" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">VA 区间释放（驱动内部）</text>
<path d="M255.0,88.0 L300.0,88.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M500.0,88.0 L525.4,88.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,102.0 L640.0,150.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,178.0 L640.0,206.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M560.0,220.0 L255.0,220.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,234.0 L640.0,262.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M740.0,276.0 L970.0,276.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,290.0 L640.0,318.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,346.0 L640.0,374.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M540.0,388.0 L465.0,388.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,402.0 L640.0,430.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M400.0,402.0 L400.0,430.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M320.0,458.0 L320.0,576.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M480.0,458.0 L480.0,502.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M640.0,458.0 L640.0,502.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M1040.0,290.0 L1040.0,576.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M160.0,234.0 L150.0,576.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<text x="650.0" y="304.0" font-size="8" fill="#566573">push</text>
<text x="650.0" y="360.0" font-size="8" fill="#566573">取出</text>
<path d="M720.0,530.0 L720.0,672.0" fill="none" stroke="#1c2833" stroke-width="1.5" marker-end="url(#arr)"/>
<path d="M245.9,712.0 L245.9,750.0" fill="none" stroke="#7f8c8d" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#arr)"/>
<path d="M429.7,712.0 L429.7,750.0" fill="none" stroke="#7f8c8d" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#arr)"/>
<path d="M82.0,712.0 L82.0,750.0" fill="none" stroke="#7f8c8d" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#arr)"/>
<path d="M1040.0,604.0 L1040.0,672.0" fill="none" stroke="#7f8c8d" stroke-width="1.5" stroke-dasharray="6 4" marker-end="url(#arr)"/>
<line x1="82.0" y1="672.0" x2="1040.0" y2="672.0" stroke="#34495e" stroke-width="1.2"/>
<line x1="245.9" y1="672.0" x2="245.9" y2="684.0" stroke="#34495e" stroke-width="1.2"/>
<line x1="429.7" y1="672.0" x2="429.7" y2="684.0" stroke="#34495e" stroke-width="1.2"/>
<line x1="82.0" y1="672.0" x2="82.0" y2="684.0" stroke="#34495e" stroke-width="1.2"/>
<rect x="247.0" y="66.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="255.0" y="77.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">1</text>
<rect x="492.0" y="66.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="500.0" y="77.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">2</text>
<rect x="746.6" y="66.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="754.6" y="77.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">3</text>
<rect x="753.2" y="142.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="761.2" y="153.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">4</text>
<rect x="712.0" y="198.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="720.0" y="209.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">5</text>
<rect x="247.0" y="198.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="255.0" y="209.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">6</text>
<rect x="217.0" y="568.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="225.0" y="579.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">6</text>
<rect x="730.0" y="254.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="740.0" y="265.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">7a</text>
<rect x="730.0" y="366.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="740.0" y="377.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">7b</text>
<rect x="1102.0" y="254.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="1110.0" y="265.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">8</text>
<rect x="1067.0" y="568.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="1075.0" y="579.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">8</text>
<rect x="457.0" y="366.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="465.0" y="377.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">9</text>
<rect x="502.0" y="422.5" width="16.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="510.0" y="433.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">9</text>
<rect x="388.3" y="568.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="398.3" y="579.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">9a</text>
<rect x="114.0" y="676.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="124.0" y="687.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">9b</text>
<rect x="284.5" y="676.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="294.5" y="687.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">9c</text>
<rect x="718.0" y="422.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="728.0" y="433.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">10</text>
<rect x="481.5" y="676.5" width="20.0" height="15" rx="7.5" fill="#1c2833" stroke="#ffffff" stroke-width="1"/>
<text x="491.5" y="687.5" font-size="9" font-weight="bold" text-anchor="middle" fill="#ffffff">10</text>
<text x="10.0" y="800.0" font-size="10" font-weight="bold" fill="#34495e">例子：起始就是碎片</text>
<text x="205.0" y="800.0" font-size="10" font-weight="bold" text-anchor="middle" fill="#34495e">第 1 遍：8</text>
<text x="722.5" y="800.0" font-size="10" font-weight="bold" text-anchor="middle" fill="#34495e">第 2 遍：9b、9c、9b、9c</text>
<rect x="10.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="18.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">起始（碎片）</text>
<text x="89.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="169.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="16.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="16.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="34.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="49.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V0</text>
<rect x="74.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="89.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="114.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="129.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V2</text>
<rect x="154.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="169.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V3</text>
<rect x="34.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="51.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="74.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="91.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="114.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="131.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="154.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="171.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H3</text>
<line x1="49.0" y1="880.0" x2="51.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="89.0" y1="880.0" x2="91.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="129.0" y1="880.0" x2="131.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="169.0" y1="880.0" x2="171.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="18.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 8 MB</text>
<text x="18.0" y="944.0" font-size="9.5" fill="#1c2833">在用的 H1 夹在两段空闲之间</text>
<text x="18.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<text x="18.0" y="1004.0" font-size="9" fill="#7f8c8d">空</text>
<line x1="193.0" y1="878.0" x2="217.0" y2="878.0" stroke="#1c2833" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="205.0" y="872.0" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#1c2833">8</text>
<rect x="217.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="225.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">S1</text>
<text x="296.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="376.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="223.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="223.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="241.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="256.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V0</text>
<rect x="281.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="296.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="321.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="336.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V2</text>
<rect x="361.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="241.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="258.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="281.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="298.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="321.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="338.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="361.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<line x1="256.0" y1="880.0" x2="258.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="296.0" y1="880.0" x2="298.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="336.0" y1="880.0" x2="338.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="225.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 6 MB</text>
<text x="225.0" y="944.0" font-size="9.5" fill="#1c2833">H3 的 HBM 归还</text>
<text x="225.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<rect x="225.0" y="988.0" width="36.0" height="18.0" rx="3" fill="#fef9e7" stroke="#b7950b" stroke-width="0.8"/>
<text x="243.0" y="1001.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7d6608">V0</text>
<rect x="269.0" y="988.0" width="36.0" height="18.0" rx="3" fill="#fef9e7" stroke="#b7950b" stroke-width="0.8"/>
<text x="287.0" y="1001.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7d6608">V2</text>
<line x1="400.0" y1="878.0" x2="424.0" y2="878.0" stroke="#1c2833" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="412.0" y="872.0" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#1c2833">9b</text>
<rect x="424.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="432.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">S2</text>
<text x="503.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="583.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="430.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="430.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="448.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="463.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V0</text>
<rect x="488.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="503.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="528.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="543.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V2</text>
<rect x="568.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="448.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="465.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H0</text>
<rect x="488.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="505.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="528.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="545.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="568.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<line x1="503.0" y1="880.0" x2="505.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="543.0" y1="880.0" x2="545.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="432.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 6 MB</text>
<text x="432.0" y="944.0" font-size="9.5" fill="#1c2833">V0 变成未映射的空洞</text>
<text x="432.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<rect x="432.0" y="988.0" width="36.0" height="18.0" rx="3" fill="#fef9e7" stroke="#b7950b" stroke-width="0.8"/>
<text x="450.0" y="1001.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7d6608">V2</text>
<line x1="607.0" y1="878.0" x2="631.0" y2="878.0" stroke="#1c2833" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="619.0" y="872.0" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#1c2833">9c</text>
<rect x="631.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="639.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">S3</text>
<text x="710.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="790.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="637.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="637.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="655.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="670.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V0</text>
<rect x="695.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="710.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="735.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="750.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V2</text>
<rect x="775.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="655.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<rect x="695.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="712.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="735.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="752.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="775.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<line x1="710.0" y1="880.0" x2="712.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="750.0" y1="880.0" x2="752.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="639.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 4 MB</text>
<text x="639.0" y="944.0" font-size="9.5" fill="#1c2833">H0 的 HBM 归还</text>
<text x="639.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<rect x="639.0" y="988.0" width="36.0" height="18.0" rx="3" fill="#fef9e7" stroke="#b7950b" stroke-width="0.8"/>
<text x="657.0" y="1001.0" font-size="9" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7d6608">V2</text>
<line x1="814.0" y1="878.0" x2="838.0" y2="878.0" stroke="#1c2833" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="826.0" y="872.0" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#1c2833">9b</text>
<rect x="838.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="846.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">S4</text>
<text x="917.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="997.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="844.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="844.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="862.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="877.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V0</text>
<rect x="902.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="917.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="942.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="957.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V2</text>
<rect x="982.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="862.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<rect x="902.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="919.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="942.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#d5d8dc" stroke="#7f8c8d" stroke-width="0.8"/>
<text x="959.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1c2833">H2</text>
<rect x="982.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<line x1="917.0" y1="880.0" x2="919.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="846.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 4 MB</text>
<text x="846.0" y="944.0" font-size="9.5" fill="#1c2833">V2 变成未映射的空洞</text>
<text x="846.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<text x="846.0" y="1004.0" font-size="9" fill="#7f8c8d">空</text>
<line x1="1021.0" y1="878.0" x2="1045.0" y2="878.0" stroke="#1c2833" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="1033.0" y="872.0" font-size="8.5" font-weight="bold" text-anchor="middle" fill="#1c2833">9c</text>
<rect x="1045.0" y="818.0" width="183.0" height="200.0" rx="4" fill="#fbfcfc" stroke="#d5d8dc" stroke-width="0.8"/>
<text x="1053.0" y="836.0" font-size="10.5" font-weight="bold" fill="#1c2833">S5 结束</text>
<text x="1124.0" y="850.0" font-size="8" text-anchor="middle" fill="#2e86c1">段 E（PyTorch 管）</text>
<text x="1204.0" y="850.0" font-size="8" text-anchor="middle" fill="#7d3c98">驱动管</text>
<text x="1051.0" y="872.0" font-size="8.5" fill="#566573">VA</text>
<text x="1051.0" y="904.0" font-size="8.5" fill="#566573">HBM</text>
<rect x="1069.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="1084.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V0</text>
<rect x="1109.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="1124.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">V1</text>
<rect x="1149.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="1164.0" y="869.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#7f8c8d">V2</text>
<rect x="1189.0" y="856.0" width="30.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="1069.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<rect x="1109.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="1126.0" y="901.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="1149.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<rect x="1189.0" y="888.0" width="34.0" height="24.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<line x1="1124.0" y1="880.0" x2="1126.0" y2="887.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="1053.0" y="928.0" font-size="9.5" font-weight="bold" fill="#1c2833">reserved = 2 MB</text>
<text x="1053.0" y="944.0" font-size="9.5" fill="#1c2833">段还在（H1 在用），不删段</text>
<text x="1053.0" y="982.0" font-size="8.5" fill="#b7950b">to_unmap</text>
<text x="1053.0" y="1004.0" font-size="9" fill="#7f8c8d">空</text>
<line x1="401.0" y1="796.0" x2="401.0" y2="1018.0" stroke="#c0392b" stroke-width="1.2" stroke-dasharray="4 3"/>
<text x="10.0" y="1032.0" font-size="11" fill="#1c2833">cudaFree：HBM + VA　　cuMemRelease：HBM　　cuMemAddressFree：VA　　cuMemUnmap：都不释放</text>
<text x="10.0" y="1050.0" font-size="11" fill="#1c2833">只处理池里的空闲块，在用的 H1 不动。</text>
<text x="10.0" y="1068.0" font-size="11" fill="#1c2833">结果：V0、V2 是不相邻的空洞，4 MB 连续请求用不了。</text>
<text x="10.0" y="1100.0" font-size="11" font-weight="bold" fill="#1c2833">颜色：</text>
<rect x="56.0" y="1088.0" width="28.0" height="15.0" rx="4" fill="#fadbd8" stroke="#c0392b" stroke-width="1"/>
<text x="90.0" y="1100.0" font-size="11" fill="#1c2833">释放</text>
<rect x="146.0" y="1088.0" width="28.0" height="15.0" rx="4" fill="#eaecee" stroke="#7f8c8d" stroke-width="1"/>
<text x="180.0" y="1100.0" font-size="11" fill="#1c2833">查询</text>
<rect x="236.0" y="1088.0" width="28.0" height="15.0" rx="4" fill="#fef9e7" stroke="#b7950b" stroke-width="1"/>
<text x="270.0" y="1100.0" font-size="11" fill="#1c2833">存储</text>
<rect x="326.0" y="1088.0" width="28.0" height="15.0" rx="4" fill="#f8f9f9" stroke="#aab7b8" stroke-width="1"/>
<text x="360.0" y="1100.0" font-size="11" fill="#1c2833">内部</text>
<rect x="416.0" y="1088.0" width="28.0" height="15.0" rx="4" fill="#ffffff" stroke="#aab7b8" stroke-width="1" stroke-dasharray="4 3"/>
<text x="450.0" y="1100.0" font-size="11" fill="#1c2833">不用到</text>
<text x="10.0" y="1122.0" font-size="11" fill="#566573">虚线 = 隐式（驱动内部）。cudaFree 内部调用的 cuMem* 未知，需 trace 才能确定。</text>
<text x="10.0" y="1137.0" font-size="11" fill="#566573">没画：small 池、graph pool 分支、cudaDeviceSynchronize 分支、GMLake 的 GC。第 5 层是通用知识，未核对。</text>
</svg>
```
