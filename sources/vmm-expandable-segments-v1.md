# VMM 与 expandable_segments 的关系

从 kernel 的角度，端到端展示 VMM 的流程，并说明 VMM 为什么是 expandable_segments 段尾扩展的关键。

新 session 只需读本 paste。

- 基于：无（首版草稿，尚未发布）
- 对应文件：`~/diagrams/vmm_expend_explain/vmm_expend_explain.svg`（源文件）

## 核心论点

**一句话目的（草稿，用户未修改）：** 从 kernel 的角度端到端展示 VMM 的流程，说明 VMM 是 expandable_segments 段尾扩展的关键。

**核心知识点（用户原话，已润色）：** VMM 把虚拟地址和物理 chunk 解耦：先预留一段连续 VA，再逐个映射 chunk，在预留范围内保证虚拟地址段连续。expand 只在段尾映射新 chunk，地址接着往后走，因此段保持连续，物理 chunk 不需要连续。

**整张图的精髓：** 连续的是虚拟地址，不是物理 chunk。推理链：cudaMalloc 的地址不保证相邻（K18）→ expand 要求新地址紧接旧地址（K12）→ 因此要先用 VMM 预留连续 VA（K1、K19）→ 段尾用 C 映射新 chunk（K3、K12）。

**措辞说明：** "VMM"指 CUDA 低级虚拟内存管理 API 加上驱动维护的页表。"连续"指虚拟地址连续，没有说物理上连续。用户原话里的"永远保证"改为"在预留范围内保证"：段连续依赖预留的 VA 范围足够大，超出预留范围就不能靠段尾映射继续扩展。

## 术语

| 术语 | 含义 |
|---|---|
| VMM | CUDA 低级虚拟内存管理 API（A–G），把虚拟地址和物理显存分开管理 |
| expandable_segments | PyTorch 的分配模式，每个池占用一段连续 VA，需要时在段尾追加映射 |
| VA | 虚拟地址。kernel 看到的地址，段内连续 |
| chunk | 2MB 物理显存块，位置可以散落 |
| PTE / 页表 | 驱动写入的映射表项，记录 VA 槽指向哪个 chunk，由 GPU 硬件查询 |
| expand | 在段尾新建 chunk、映射、设权限，段变长，前面的映射不动 |
| A–G | cuMemAddressReserve、cuMemCreate、cuMemMap、cuMemSetAccess、cuMemUnmap、cuMemAddressFree、cuMemRelease |

## 背景

- 起因：读 GMLake 源码和论文时，对 VMM 与 expandable_segments 的关系不清楚，于是画图梳理。
- 图中没有实测数字，所有数字（6/8 格、chunk 编号、柱高）都是示意。

## 知识

| ID | 内容 | 来源 | 状态 | 确认 | 你的理解 | 对应图元 |
|---|---|---|---|---|---|---|
| K1 | CUDA 低级 VMM 调用 A–G：A cuMemAddressReserve 预留 VA；B cuMemCreate 创建物理 chunk；C cuMemMap 映射到 VA；D cuMemSetAccess 设权限；E cuMemUnmap 解除映射；F cuMemAddressFree 释放 VA；G cuMemRelease 释放物理 chunk | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:279,236,351,185,304,306,243（调用位置已读） | verified（调用位置） | 待确认 | 先预留一段连续 VA，再逐个映射 chunk，在预留范围内保证虚拟地址段连续（用户原话） | 图中 A–G 紫框 |
| K2 | 物理 chunk 粒度 2MB（granularitySize = 2097152） | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:104 | verified | 待确认 |  | chunk 层 |
| K3 | 每个 chunk 的映射：先 cuMemMap 再 setMemAccess，逐 chunk 一次 | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:351–352 | verified | 待确认 |  | C、D 紫框；页表层箭头 |
| K4 | cuMemUnmap 按整段 VA 调用一次（VirDevPtr 析构：先 unmap 整段，再 AddressFree） | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:300–310 | verified | 待确认 |  | E、F 紫框 |
| K5 | 新分配路径：PhyBlock 构造调用 cuMemCreate（每 chunk 一次）；拼接构造只映射、不调用 cuMemCreate | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:236、410–420 | verified | 待确认 |  | B 紫框（未画拼接路径） |
| K6 | 一个物理 chunk 可被多个 Block 映射（PhyBlock.mapped_blocks） | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:264 | verified | 待确认 |  | 未画入（图已不画共享 chunk） |
| K7 | 占用不重叠靠 PhyBlock.free 标记；取用已占用 chunk 时程序 exit | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:222；GMLake/src/CUDACachingAllocator.cpp:2309–2312 | verified | 待确认 |  | 未画入 |
| K8 | GMLake 拼接候选按 chunk 大小从大到小贪心累加 | glake-master.zip：GMLake/src/CUDACachingAllocator.cpp:2767 | verified | 待确认 |  | 未画入（GMLake 策略不在本图） |
| K9 | StitchFree 只回收拼接块的 VA 映射；物理 chunk 仍由其他持有者引用 | glake-master.zip：GMLake/src/CUDACachingAllocator.cpp:2554–2702 | inferred（读了路径，未运行） | 待确认 |  | 未画入 |
| K10 | cuMemRelease（G）在 PhyBlock 最后一个引用消失时调用（shared_ptr 析构） | glake-master.zip：GMLake/include/cuda_vmm_allocator.h:239–244 | inferred | 待确认 |  | G 紫框；灰色"已释放"格 |
| K11 | GMLake 补丁里没有 expandable_segments 相关代码（grep 为 0） | glake-master.zip：GMLake/src、GMLake/include 的 grep | verified | 待确认 |  | 未画入（背景） |
| K12 | expandable_segments 段尾 expand：预留较大 VA，需要时在段尾新建 chunk（B）、映射（C）、设权限（D），前面映射不动 | PyTorch 通用设计，来自记忆 | unverified | 待确认 |  | expand 小图 |
| K13 | PyTorch 内部 fork 注释："We use cuda's low-level memory APIs, which are similar to mmap, to extend the..." | 内部 fbsource c10/cuda/CUDACachingAllocator.cpp:361–363（注释存在，完整实现未读） | verified（注释存在） | 待确认 |  | （背景） |
| K14 | 页表由驱动维护（cuMemMap 写入 PTE）；GPU 硬件在访存时查页表完成 VA→PA | 通用 NVIDIA GPU 机制，仓库未包含 | unverified | 待确认 |  | 页表层、GMMU 箭头 |
| K15 | GPU 访存路径：SM（LD/ST、L1）→ TLB → GMMU（页表遍历）→ L2 → 内存控制器 → HBM；TLB 命中则跳过页表遍历 | 通用 NVIDIA GPU 结构示意，仓库未包含 | unverified | 待确认 |  | GPU 框与 ①–⑤ 箭头 |
| K16 | 600MB 段 = 300 个 2MB chunk；图中 6/8 格与 chunk 编号为示意 | 算术：600 ÷ 2 = 300 | verified（算术） | 待确认 |  | VA 段 |
| K17 | 驱动在 unmap 时使 TLB 失效 | 通用做法，未核实 | unverified | 待确认 |  | driver 框 |
| K18 | cudaMalloc 返回的地址不保证与已有段相邻，由驱动决定 | CUDA 通用行为，未核实 | unverified | 待确认 |  | 顶部简介 |
| K19 | VMM 是 expandable_segments 段尾扩展的前提：先预留连续 VA，之后在段尾映射新 chunk，地址保持连续 | 由逻辑推出，与 K13 注释一致 | inferred | 待确认 |  | 顶部简介；expand 小图 |

状态：verified（读过源码或注释确认）/ inferred（由代码或逻辑推出）/ unverified（通用机制或记忆，未核实）。确认：待确认 / 已确认。

## 图的内容

按从上到下的顺序看：

1. **顶部简介（六行）：** VMM 是什么、expandable_segments 是什么、为什么 VMM 是关键、图的读法。
2. **GPU（硬件）框：** SM（LD/ST、L1）→ TLB → GMMU → L2 → 内存控制器 → HBM，对应 ①–⑤ 访存路径。驱动（紫色）负责 cuMemMap 写页表，unmap 时使 TLB 失效。
3. **kernel 看到的虚拟地址段 X：** 8 格，前 6 格已映射，后 2 格由 expand 新增。A 预留整段 VA；右侧 E、F 分别解除映射、释放 VA。B+C+D 的操作框在段尾两格上方，箭头向下指向这两格。
4. **HBM 的页表层：** 8 个 PTE（槽 i → chunk k）。C 写入映射，D 设权限，竖箭头把每个 PTE 连到它指向的 chunk。
5. **HBM 的 chunk 层：** 2MB chunk 散落排列。B 新建 chunk，G 释放 chunk，灰色"×"是已释放的格子，其他灰格是其他 tensor 的 chunk。访存路径的 ⑥ 从 GMMU 指向 chunk 9。
6. **底部 expand 小图：** ① 扩展前（段有 6 格）→ ② 段尾追加（B+C+D 指向第 6、7 格，箭头向下）→ ③ 扩展后（段 8 格，新 chunk #1、#9）。

## 结构

- 分组：`group-gpu`（GPU 硬件）、`group-hbm`（HBM）、`group-va`（kernel 看到的虚拟地址段）
- 实体：`node-op-a` … `node-op-g`（A–G 紫框），`node-va-slot-0…5`（VA 格），`node-pte-0…5`（页表项），`node-chunk-*`（chunk），`node-sm / node-ldst / node-tlb / node-gmmu / node-l2 / node-mc / node-driver`（硬件与驱动），`node-expand-step1…3`（expand 小图）
- 关系：`edge-*`（28 组），例如 `edge-va-slot-0-pte-0`、`edge-pte-0-chunk-2`、`edge-tlb-gmmu`
- 形状：圆角方框 = 实体；虚线方框 = 未映射或已释放；箭头 = 调用或映射
- 颜色：紫色 = CUDA 驱动调用与驱动；灰色 = 硬件；绿色 = kernel 视图的 VA 段；彩色 = 同一颜色的 VA 格、PTE、chunk 对应同一条映射
- 阅读顺序：自上而下。先看简介，再看 GPU 硬件，然后是 kernel 的 VA 段，接着是 HBM 的页表层和 chunk 层，最后看底部 expand 小图

## SVG 约定

- 尺寸：`viewBox="0 0 1200 1284"`
- id 规则：当前生成的是 `node-<slug>`、`node-<slug>-label-<n>`、`edge-<from>-<to>`、`group-<slug>`；还没有按 skill 的 `group-panel-<name>` 规则重命名
- `data-k` 标在方框、连线上，不在 `<g>` 上
- 字体：`'PingFang SC', 'Hiragino Sans GB', 'Helvetica Neue', Arial`（skill 要求 Noto Sans CJK SC，未改）
- 坐标是手排或用脚本生成的，没有用布局引擎

## 修改记录

- 2026-10-27 生成草稿：标题改为 VMM 与 expandable_segments 的关系；加入顶部简介；expand 小图的箭头改为向下；去掉 GMLake 字样（session unknown）
- 2026-10-27 用户确认核心知识点（改为"在预留范围内保证"），发布首版 paste（基于：无；session unknown）

## 未解决

- [ ] 第 1 组（K1–K4）的检查题"为什么 A 和 C 要分两步"还没回答，K1–K4 仍待确认
- [ ] 第 2–5 组知识（K5–K19）还没有逐组确认，全部待确认
- [ ] 通用机制 K14、K15、K17、K18 未核实；K12 来自记忆
- [ ] SVG 的 id 和 data-k 还没有按 skill 的面板规则（group-panel-*，data-k 放在 <g> 上）
- [ ] PNG 未渲染（本机没有 node 和 npm）

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1284" viewBox="0 0 1200 1284" font-family="'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="arrp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#6c3483"/>
    </marker>
  </defs>
  <rect width="1200" height="1284" fill="#ffffff"/>

  <text x="40" y="46" font-size="24" font-weight="bold" fill="#222">VMM 与 expandable_segments 的关系</text>
  <text x="40" y="72" font-size="14" fill="#555">从 kernel 的角度，端到端展示 VMM 的流程；expandable_segments 是 VMM 的一种用法</text>
  <text id="node-box-40-100-label-1" x="40" y="104" font-size="13" fill="#333333">「VMM」是 CUDA 的低级虚拟内存管理 API（A–G）：先预留一段连续地址（A），再把物理 chunk 逐个映射到地址上（C）。</text>
  <text id="node-box-40-100-label-3" x="40" y="126" font-size="13" fill="#333333">「expandable_segments」是 PyTorch 的分配模式：每个池占用一段连续地址，需要显存时在段尾追加映射，地址始终连续。</text>
  <text id="node-box-40-100-label-4" x="40" y="148" font-size="13" fill="#333333">为什么 VMM 是关键：expand 要求新地址紧接在旧地址后面；cudaMalloc 拿到的地址不保证相邻。</text>
  <text id="node-box-40-100-label-5" x="40" y="170" font-size="13" fill="#333333">VMM 先预留连续地址，再在段尾映射新 chunk，所以地址连续，而物理 chunk 可以不连续。</text>
  <text id="node-box-40-100-label-8" x="40" y="192" font-size="13" fill="#333333">图的读法：从 kernel 的角度 e2e 展示 VMM 流程。上方是 kernel 看到的连续 VA 段，下方是页表和物理 chunk。</text>
  <text id="node-box-40-100-label-9" x="40" y="214" font-size="13" fill="#333333">底部 expand 小图是段尾追加的一步。</text>

  <!-- ===== GPU hardware components ===== -->
  <g transform="translate(0,140)">
  <rect id="node-box-40-100" x="40" y="100" width="1120" height="240" rx="12" fill="#f7f7f7" stroke="#555" stroke-width="1.5"/>
  <text id="node-box-40-100-label-2" x="56" y="120" font-size="13" font-weight="bold" fill="#333">GPU（硬件）</text>

  <rect id="node-sm" data-k="K15" x="60" y="136" width="190" height="126" rx="8" fill="#ffffff" stroke="#555" stroke-width="1.5"/>
  <text id="node-sm-label-1" x="72" y="156" font-size="12" font-weight="bold" fill="#222">SM 流多处理器</text>
  <rect id="node-ldst" data-k="K15" x="76" y="168" width="158" height="32" rx="5" fill="#eeeeee" stroke="#888"/>
  <text id="node-ldst-label-1" x="155" y="189" font-size="12" text-anchor="middle" fill="#222">LD/ST 单元</text>
  <rect id="node-l1" data-k="K15" x="76" y="212" width="158" height="34" rx="5" fill="#eeeeee" stroke="#888"/>
  <text id="node-l1-label-1" x="155" y="234" font-size="12" text-anchor="middle" fill="#222">L1 数据缓存</text>

  <rect id="node-tlb" data-k="K15" x="290" y="150" width="150" height="60" rx="8" fill="#eeeeee" stroke="#888" stroke-width="1.5"/>
  <text id="node-tlb-label-1" x="365" y="174" font-size="12" font-weight="bold" text-anchor="middle" fill="#222">TLB</text>
  <text id="node-tlb-label-2" x="365" y="194" font-size="11" text-anchor="middle" fill="#222">缓存 VA → PA</text>

  <rect id="node-gmmu" data-k="K15" x="480" y="136" width="160" height="80" rx="8" fill="#eeeeee" stroke="#888" stroke-width="1.5"/>
  <text id="node-gmmu-label-1" x="560" y="162" font-size="13" font-weight="bold" text-anchor="middle" fill="#222">GMMU</text>
  <text id="node-gmmu-label-2" x="560" y="182" font-size="11" text-anchor="middle" fill="#222">页表遍历 · 权限检查</text>

  <rect id="node-l2" data-k="K15" x="660" y="270" width="140" height="60" rx="8" fill="#eeeeee" stroke="#888" stroke-width="1.5"/>
  <text id="node-l2-label-1" x="730" y="296" font-size="12" font-weight="bold" text-anchor="middle" fill="#222">L2 缓存</text>
  <text id="node-l2-label-2" x="730" y="314" font-size="11" text-anchor="middle" fill="#222">物理地址索引</text>

  <rect id="node-mc" data-k="K15" x="960" y="270" width="140" height="60" rx="8" fill="#eeeeee" stroke="#888" stroke-width="1.5"/>
  <text id="node-mc-label-1" x="1030" y="296" font-size="12" font-weight="bold" text-anchor="middle" fill="#222">内存控制器</text>
  <text id="node-mc-label-2" x="1030" y="314" font-size="11" text-anchor="middle" fill="#222">→ HBM</text>

  <rect id="node-driver" data-k="K1,K14" x="880" y="136" width="220" height="64" rx="8" fill="#d2b4de" stroke="#6c3483" stroke-width="1.5"/>
  <text id="node-driver-label-1" x="990" y="158" font-size="12" font-weight="bold" text-anchor="middle" fill="#4a235a">CUDA 驱动（软件）</text>
  <text id="node-driver-label-2" x="990" y="176" font-size="11" text-anchor="middle" fill="#4a235a">发出 A–G 这些调用</text>
  <text id="node-driver-label-3" x="990" y="192" font-size="11" text-anchor="middle" fill="#4a235a">unmap 时使 TLB 失效</text>

  <!-- Access path inside GPU -->
  <line id="edge-ldst-tlb" data-k="K15" x1="234" y1="184" x2="290" y2="184" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-box-40-100-label-7" x="256" y="176" font-size="12" font-weight="bold" fill="#333">①</text>
  <line id="edge-tlb-gmmu" data-k="K15" x1="440" y1="180" x2="480" y2="180" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-box-40-100-label-6" x="452" y="172" font-size="12" font-weight="bold" fill="#333">②</text>
  <line id="edge-gmmu-l2" data-k="K15" x1="620" y1="216" x2="660" y2="270" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-box-40-100-label-10" x="646" y="248" font-size="12" font-weight="bold" fill="#333">④</text>
  <line id="edge-l2-mc" data-k="K15" x1="800" y1="300" x2="960" y2="300" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-box-40-100-label-11" x="872" y="292" font-size="12" font-weight="bold" fill="#333">⑤</text>

  <!-- Walk arrow: GMMU reads the PTE for slot 2 -->
  <line id="edge-gmmu-pte-2" data-k="K14" x1="590" y1="216" x2="644" y2="478" stroke="#333" stroke-width="1.5" stroke-dasharray="5 4" marker-end="url(#arr)"/>
  <text id="node-box-40-100-label-12" x="520" y="300" font-size="12" font-weight="bold" fill="#333">③ 读 PTE</text>

  <!-- ===== kernel-visible VA segment (between GPU and HBM) ===== -->
  <text x="280" y="362" font-size="12" font-weight="bold" fill="#225c33">kernel 看到的虚拟地址段 X（示意 6 格）</text>

  <!-- A: reserve the VA segment -->
  <rect id="node-op-a" data-k="K1" x="40" y="352" width="210" height="40" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-a-label-1" x="52" y="368" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">A cuMemAddressReserve</text>
  <text id="node-op-a-label-2" x="52" y="384" font-size="11" fill="#3d2466">预留连续 VA（一次）</text>
  <line id="edge-op-a-va-slot-0" data-k="K14" x1="250" y1="372" x2="280" y2="382" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <rect id="node-va-slot-0" data-k="K1,K16" x="280" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-0-label-1" x="338" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 0</text>
  <rect id="node-va-slot-1" data-k="K1,K16" x="397" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-1-label-1" x="455" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 1</text>
  <rect id="node-va-slot-2" data-k="K1,K16" x="514" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-2-label-1" x="572" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 2</text>
  <rect id="node-va-slot-3" data-k="K1,K16" x="631" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-3-label-1" x="689" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 3</text>
  <rect id="node-va-slot-4" data-k="K1,K16" x="748" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-4-label-1" x="806" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 4</text>
  <rect id="node-va-slot-5" data-k="K1,K16" x="865" y="370" width="117" height="44" fill="#7fcf94" stroke="#225c33" stroke-width="1.5"/>
  <text id="node-va-slot-5-label-1" x="923" y="397" font-size="12" font-weight="bold" text-anchor="middle" fill="#173f25">槽 5</text>

  <!-- E / F: release the VA segment -->
  <rect id="node-op-e" data-k="K1,K4" x="1006" y="362" width="124" height="32" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-e-label-1" x="1012" y="377" font-size="10" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">E cuMemUnmap</text>
  <text id="node-op-e-label-2" x="1012" y="389" font-size="10" fill="#3d2466">释放段时整段解除</text>
  <line id="edge-op-e-va-slot-5" data-k="K14" x1="1006" y1="378" x2="982" y2="378" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <rect id="node-op-f" data-k="K1,K4" x="1006" y="400" width="124" height="32" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-f-label-1" x="1012" y="415" font-size="10" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">F cuMemAddressFree</text>
  <text id="node-op-f-label-2" x="1012" y="427" font-size="10" fill="#3d2466">释放 VA</text>
  <line id="edge-op-f-va-slot-5" data-k="K14" x1="1006" y1="416" x2="982" y2="408" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <!-- ===== HBM: page table layer + physical chunk layer ===== -->
  <rect id="group-hbm" data-k="K14" x="40" y="440" width="1120" height="320" rx="12" fill="#e8e8e8" stroke="#333" stroke-width="1.5"/>
  <text id="node-group-hbm-label-1" x="56" y="460" font-size="13" font-weight="bold" fill="#333">HBM（显存：页表 + 数据）</text>

  <!-- C: map each chunk into a slot (writes PTE) -->
  <rect id="node-op-c" data-k="K1,K3" x="56" y="478" width="200" height="36" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-c-label-1" x="68" y="494" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">C cuMemMap</text>
  <text id="node-op-c-label-2" x="68" y="508" font-size="11" fill="#3d2466">每个 chunk 一次 · 写页表</text>
  <line id="edge-op-c-pte-0" data-k="K3,K14" x1="256" y1="496" x2="280" y2="496" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <!-- D: set access on each mapping -->
  <rect id="node-op-d" data-k="K1,K3" x="56" y="526" width="200" height="36" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-d-label-1" x="68" y="542" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">D cuMemSetAccess</text>
  <text id="node-op-d-label-2" x="68" y="556" font-size="11" fill="#3d2466">每条映射设置权限</text>
  <line id="edge-op-d-pte-0" data-k="K3,K14" x1="256" y1="544" x2="290" y2="522" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <!-- Page table entries -->
  <rect id="node-pte-0" data-k="K3,K14" x="280" y="478" width="72" height="44" rx="5" fill="#aed6f1" stroke="#2874a6" stroke-width="1.5"/>
  <text id="node-pte-0-label-1" x="316" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#1b4f72">槽 0</text>
  <text id="node-pte-0-label-2" x="316" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#1b4f72">→ chunk 2</text>
  <rect id="node-pte-1" data-k="K3,K14" x="444" y="478" width="72" height="44" rx="5" fill="#fad7a0" stroke="#b9770e" stroke-width="1.5"/>
  <text id="node-pte-1-label-1" x="480" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#7e5109">槽 1</text>
  <text id="node-pte-1-label-2" x="480" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#7e5109">→ chunk 5</text>
  <rect id="node-pte-2" data-k="K3,K14" x="608" y="478" width="72" height="44" rx="5" fill="#d2b4de" stroke="#6c3483" stroke-width="2"/>
  <text id="node-pte-2-label-1" x="644" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#4a235a">槽 2</text>
  <text id="node-pte-2-label-2" x="644" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#4a235a">→ chunk 9</text>
  <rect id="node-pte-3" data-k="K3,K14" x="772" y="478" width="72" height="44" rx="5" fill="#a3e4d7" stroke="#117a65" stroke-width="1.5"/>
  <text id="node-pte-3-label-1" x="808" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#0e6655">槽 3</text>
  <text id="node-pte-3-label-2" x="808" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#0e6655">→ chunk 10</text>
  <rect id="node-pte-4" data-k="K3,K14" x="936" y="478" width="72" height="44" rx="5" fill="#f9e79f" stroke="#b7950b" stroke-width="1.5"/>
  <text id="node-pte-4-label-1" x="972" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#7d6608">槽 4</text>
  <text id="node-pte-4-label-2" x="972" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#7d6608">→ chunk 13</text>
  <rect id="node-pte-5" data-k="K3,K14" x="1018" y="478" width="72" height="44" rx="5" fill="#f5b7b1" stroke="#922b21" stroke-width="1.5"/>
  <text id="node-pte-5-label-1" x="1054" y="496" font-size="12" font-weight="bold" text-anchor="middle" fill="#641e16">槽 5</text>
  <text id="node-pte-5-label-2" x="1054" y="512" font-size="11" font-family="Menlo, Consolas, monospace" text-anchor="middle" fill="#641e16">→ chunk 0</text>

  <!-- VA slot -> PTE mapping arrows -->
  <line id="edge-va-slot-0-pte-0" data-k="K3,K14" x1="338" y1="414" x2="316" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-va-slot-1-pte-1" data-k="K3,K14" x1="455" y1="414" x2="480" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-va-slot-2-pte-2" data-k="K3,K14" x1="572" y1="414" x2="644" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-va-slot-3-pte-3" data-k="K3,K14" x1="689" y1="414" x2="808" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-va-slot-4-pte-4" data-k="K3,K14" x1="806" y1="414" x2="972" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-va-slot-5-pte-5" data-k="K3,K14" x1="923" y1="414" x2="1054" y2="478" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>

  <!-- PTE -> chunk mapping arrows -->
  <line id="edge-pte-0-chunk-2" data-k="K3,K14" x1="316" y1="522" x2="316" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-pte-1-chunk-5" data-k="K3,K14" x1="480" y1="522" x2="480" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-pte-2-chunk-9" data-k="K3,K14" x1="644" y1="522" x2="644" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-pte-3-chunk-10" data-k="K3,K14" x1="808" y1="522" x2="808" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-pte-4-chunk-13" data-k="K3,K14" x1="972" y1="522" x2="972" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>
  <line id="edge-pte-5-chunk-0" data-k="K3,K14" x1="1054" y1="522" x2="1054" y2="600" stroke="#333" stroke-width="1.3" marker-end="url(#arr)"/>

  <text id="node-group-hbm-label-2" x="56" y="620" font-size="12" font-weight="bold" fill="#444">物理 chunk</text>
  <text id="node-group-hbm-label-3" x="56" y="636" font-size="11" fill="#444">2MB，位置散落</text>

  <!-- Physical chunks -->
  <rect id="node-chunk-2" data-k="K2,K6" x="280" y="600" width="72" height="44" rx="4" fill="#f5b7b1" stroke="#922b21" stroke-width="1.5"/>
  <text id="node-chunk-2-label-1" x="316" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#641e16">chunk 2</text>
  <rect id="node-chunk-other" data-k="K2" x="362" y="600" width="72" height="44" rx="4" fill="#e0e0e0" stroke="#aaa"/>
  <text id="node-chunk-other-label-1" x="398" y="627" font-size="12" text-anchor="middle" fill="#666">他用</text>
  <rect id="node-chunk-5" data-k="K2,K6" x="444" y="600" width="72" height="44" rx="4" fill="#fad7a0" stroke="#b9770e" stroke-width="1.5"/>
  <text id="node-chunk-5-label-1" x="480" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#7e5109">chunk 5</text>
  <rect id="node-chunk-released" data-k="K10" x="526" y="600" width="72" height="44" rx="4" fill="#ffffff" stroke="#999" stroke-width="1.5" stroke-dasharray="5 3"/>
  <text id="node-chunk-released-label-1" x="562" y="627" font-size="11" text-anchor="middle" fill="#999">已释放</text>
  <rect id="node-chunk-9" data-k="K2,K6" x="608" y="600" width="72" height="44" rx="4" fill="#d2b4de" stroke="#6c3483" stroke-width="2"/>
  <text id="node-chunk-9-label-1" x="644" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#4a235a">chunk 9</text>
  <rect id="node-chunk-other-2" data-k="K2" x="690" y="600" width="72" height="44" rx="4" fill="#e0e0e0" stroke="#aaa"/>
  <text id="node-chunk-other-2-label-1" x="726" y="627" font-size="12" text-anchor="middle" fill="#666">他用</text>
  <rect id="node-chunk-10" data-k="K2,K6" x="772" y="600" width="72" height="44" rx="4" fill="#a3e4d7" stroke="#117a65" stroke-width="1.5"/>
  <text id="node-chunk-10-label-1" x="808" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#0e6655">chunk 10</text>
  <rect id="node-chunk-other-3" data-k="K2" x="854" y="600" width="72" height="44" rx="4" fill="#e0e0e0" stroke="#aaa"/>
  <text id="node-chunk-other-3-label-1" x="890" y="627" font-size="12" text-anchor="middle" fill="#666">他用</text>
  <rect id="node-chunk-13" data-k="K2,K6" x="936" y="600" width="72" height="44" rx="4" fill="#f9e79f" stroke="#b7950b" stroke-width="1.5"/>
  <text id="node-chunk-13-label-1" x="972" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#7d6608">chunk 13</text>
  <rect id="node-chunk-0" data-k="K2,K6" x="1018" y="600" width="72" height="44" rx="4" fill="#f5b7b1" stroke="#922b21" stroke-width="1.5"/>
  <text id="node-chunk-0-label-1" x="1054" y="627" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" text-anchor="middle" fill="#641e16">chunk 0</text>

  <!-- B: allocate a physical chunk (new-allocation path only) -->
  <rect id="node-op-b" data-k="K1,K5" x="280" y="660" width="160" height="36" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-b-label-1" x="292" y="676" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">B cuMemCreate</text>
  <text id="node-op-b-label-2" x="292" y="690" font-size="11" fill="#3d2466">每个新 chunk 一次</text>
  <line id="edge-op-b-group-hbm" data-k="K14" x1="316" y1="660" x2="316" y2="645" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <!-- G: release a physical chunk whose last reference is gone -->
  <rect id="node-op-g" data-k="K1,K10" x="480" y="660" width="140" height="36" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-g-label-1" x="492" y="676" font-size="11" font-family="Menlo, Consolas, monospace" font-weight="bold" fill="#3d2466">G cuMemRelease</text>
  <text id="node-op-g-label-2" x="492" y="690" font-size="11" fill="#3d2466">chunk 无引用时释放</text>
  <line id="edge-op-g-group-hbm" data-k="K14" x1="562" y1="660" x2="562" y2="645" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>

  <!-- Memory controller -> chunk 9 (routed below the chunk row) -->
  <polyline id="edge-mc-route-to-chunk" data-k="K15" points="1100,300 1140,300 1140,720 644,720 644,646" fill="none" stroke="#333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-group-hbm-label-4" x="700" y="736" font-size="12" font-weight="bold" fill="#333">⑥ 读写 chunk 9</text>

  <!-- Legend -->
  <text x="40" y="1030" font-size="11" fill="#222">访存步骤：① LD/ST 发出 VA → ② TLB 未命中 → ③ GMMU 读 PTE → ④ 得到 PA，送 L2 → ⑤ L2 未命中 → 内存控制器</text>
  <text x="40" y="1048" font-size="11" fill="#222">→ ⑥ HBM 读写 chunk 9 · ⑦ 数据沿原路返回（未画）· TLB 命中则跳过 ②③ · 无映射或无权限 → 访存错误</text>
  <text x="40" y="1066" font-size="11" fill="#555">紫色框 = CUDA 驱动调用 A–G，箭头指向它作用的对象（VA 段、页表项、chunk）</text>
  <text x="40" y="1084" font-size="11" fill="#222">已有 chunk 只需映射（C、D），不需要 B；只有新建 chunk 时才调用 B。</text>

  <text x="40" y="1112" font-size="11" fill="#777">A–G 为 CUDA 低级 VMM API：A 预留 VA · B 创建 chunk · C 映射 · D 设权限 · E 解除映射 · F 释放 VA · G 释放 chunk。</text>
  <text x="40" y="1128" font-size="11" fill="#777">映射粒度为 cuMemGetAllocationGranularity 返回值（图中取 2MB）。SM / TLB / GMMU / L2 / 内存控制器为通用 NVIDIA GPU 机制示意，未经核对；图中槽位与 chunk 编号为示意。</text>

  <!-- expand example band -->
  <text x="40" y="806" font-size="14" font-weight="bold" fill="#3d2466">expand：expandable_segments 在段尾扩展</text>
  <rect id="node-expand-step1" data-k="K12,K13" x="40" y="818" width="360" height="194" rx="10" fill="#fafafa" stroke="#bbbbbb" stroke-width="1"/>
  <text id="node-expand-step1-label-1" x="52" y="840" font-size="13" font-weight="bold" fill="#222222">① expand 前</text>
  <rect id="node-box-52-904" x="52" y="904" width="40" height="30" rx="0" fill="#aed6f1" stroke="#2874a6" stroke-width="1.3"/>
  <text id="node-expand-step1-label-2" x="72.0" y="947" font-size="9" text-anchor="middle" fill="#555555">0</text>
  <rect id="node-box-92-904" x="92" y="904" width="40" height="30" rx="0" fill="#fad7a0" stroke="#b9770e" stroke-width="1.3"/>
  <text id="node-expand-step1-label-3" x="112.0" y="947" font-size="9" text-anchor="middle" fill="#555555">1</text>
  <rect id="node-box-132-904" x="132" y="904" width="40" height="30" rx="0" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
  <text id="node-expand-step1-label-4" x="152.0" y="947" font-size="9" text-anchor="middle" fill="#555555">2</text>
  <rect id="node-box-172-904" x="172" y="904" width="40" height="30" rx="0" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
  <text id="node-expand-step1-label-5" x="192.0" y="947" font-size="9" text-anchor="middle" fill="#555555">3</text>
  <rect id="node-box-212-904" x="212" y="904" width="40" height="30" rx="0" fill="#f9e79f" stroke="#b7950b" stroke-width="1.3"/>
  <text id="node-expand-step1-label-6" x="232.0" y="947" font-size="9" text-anchor="middle" fill="#555555">4</text>
  <rect id="node-box-252-904" x="252" y="904" width="40" height="30" rx="0" fill="#f5b7b1" stroke="#922b21" stroke-width="1.3"/>
  <text id="node-expand-step1-label-7" x="272.0" y="947" font-size="9" text-anchor="middle" fill="#555555">5</text>
  <rect id="node-box-292-904" x="292" y="904" width="40" height="30" rx="0" fill="#ffffff" stroke="#999999" stroke-width="1" stroke-dasharray="3 2"/>
  <text id="node-expand-step1-label-8" x="312.0" y="947" font-size="9" text-anchor="middle" fill="#555555">6</text>
  <rect id="node-box-332-904" x="332" y="904" width="40" height="30" rx="0" fill="#ffffff" stroke="#999999" stroke-width="1" stroke-dasharray="3 2"/>
  <text id="node-expand-step1-label-9" x="352.0" y="947" font-size="9" text-anchor="middle" fill="#555555">7</text>
  <text id="node-expand-step1-label-10" x="52" y="990" font-size="11" fill="#333333">段只预留地址（A），不占显存</text>
  <text id="node-expand-step1-label-11" x="52" y="1006" font-size="11" fill="#333333">后 2 格（6、7）还未映射</text>
  <rect id="node-expand-step2" data-k="K12,K13" x="420" y="818" width="360" height="194" rx="10" fill="#fafafa" stroke="#bbbbbb" stroke-width="1"/>
  <text id="node-expand-step2-label-1" x="432" y="840" font-size="13" font-weight="bold" fill="#222222">② expand：段尾追加</text>
  <rect id="node-op-b-2" data-k="K1,K5" x="432" y="848" width="336" height="36" rx="6" fill="#f5eefc" stroke="#7a4fb0" stroke-width="1.5"/>
  <text id="node-op-b-2-label-1" x="442" y="863" font-size="11" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#3d2466">B cuMemCreate ×2</text>
  <text id="node-op-b-2-label-2" x="442" y="878" font-size="11" fill="#3d2466">C cuMemMap ×2 · D cuMemSetAccess ×2</text>
  <rect id="node-box-432-904" x="432" y="904" width="40" height="30" rx="0" fill="#aed6f1" stroke="#2874a6" stroke-width="1.3"/>
  <text id="node-expand-step2-label-2" x="452.0" y="947" font-size="9" text-anchor="middle" fill="#555555">0</text>
  <rect id="node-box-472-904" x="472" y="904" width="40" height="30" rx="0" fill="#fad7a0" stroke="#b9770e" stroke-width="1.3"/>
  <text id="node-expand-step2-label-3" x="492.0" y="947" font-size="9" text-anchor="middle" fill="#555555">1</text>
  <rect id="node-box-512-904" x="512" y="904" width="40" height="30" rx="0" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
  <text id="node-expand-step2-label-4" x="532.0" y="947" font-size="9" text-anchor="middle" fill="#555555">2</text>
  <rect id="node-box-552-904" x="552" y="904" width="40" height="30" rx="0" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
  <text id="node-expand-step2-label-5" x="572.0" y="947" font-size="9" text-anchor="middle" fill="#555555">3</text>
  <rect id="node-box-592-904" x="592" y="904" width="40" height="30" rx="0" fill="#f9e79f" stroke="#b7950b" stroke-width="1.3"/>
  <text id="node-expand-step2-label-6" x="612.0" y="947" font-size="9" text-anchor="middle" fill="#555555">4</text>
  <rect id="node-box-632-904" x="632" y="904" width="40" height="30" rx="0" fill="#f5b7b1" stroke="#922b21" stroke-width="1.3"/>
  <text id="node-expand-step2-label-7" x="652.0" y="947" font-size="9" text-anchor="middle" fill="#555555">5</text>
  <rect id="node-box-672-904" x="672" y="904" width="40" height="30" rx="0" fill="#ffffff" stroke="#7a4fb0" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text id="node-box-672-904-label-1" x="692.0" y="924" font-size="14" font-weight="bold" text-anchor="middle" fill="#7a4fb0">+</text>
  <text id="node-expand-step2-label-8" x="692.0" y="947" font-size="9" text-anchor="middle" fill="#555555">6</text>
  <rect id="node-box-712-904" x="712" y="904" width="40" height="30" rx="0" fill="#ffffff" stroke="#7a4fb0" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text id="node-box-712-904-label-1" x="732.0" y="924" font-size="14" font-weight="bold" text-anchor="middle" fill="#7a4fb0">+</text>
  <text id="node-expand-step2-label-9" x="732.0" y="947" font-size="9" text-anchor="middle" fill="#555555">7</text>
  <line id="edge-op-b-2-expand-step2" data-k="K14" x1="692.0" y1="884" x2="692.0" y2="903" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>
  <line id="edge-op-b-2-expand-step2-2" data-k="K14" x1="732.0" y1="884" x2="732.0" y2="903" stroke="#7a4fb0" stroke-width="1.5" marker-end="url(#arrp)"/>
  <text id="node-expand-step2-label-10" x="432" y="990" font-size="11" fill="#333333">在段尾新建 2 个 chunk 并映射</text>
  <text id="node-expand-step2-label-11" x="432" y="1006" font-size="11" fill="#333333">前 6 格的映射不动</text>
  <rect id="node-expand-step3" data-k="K12,K13" x="800" y="818" width="360" height="194" rx="10" fill="#fafafa" stroke="#bbbbbb" stroke-width="1"/>
  <text id="node-expand-step3-label-1" x="812" y="840" font-size="13" font-weight="bold" fill="#222222">③ expand 后</text>
  <rect id="node-box-812-904" x="812" y="904" width="40" height="30" rx="0" fill="#aed6f1" stroke="#2874a6" stroke-width="1.3"/>
  <text id="node-expand-step3-label-2" x="832.0" y="947" font-size="9" text-anchor="middle" fill="#555555">0</text>
  <rect id="node-box-852-904" x="852" y="904" width="40" height="30" rx="0" fill="#fad7a0" stroke="#b9770e" stroke-width="1.3"/>
  <text id="node-expand-step3-label-3" x="872.0" y="947" font-size="9" text-anchor="middle" fill="#555555">1</text>
  <rect id="node-box-892-904" x="892" y="904" width="40" height="30" rx="0" fill="#d2b4de" stroke="#6c3483" stroke-width="1.3"/>
  <text id="node-expand-step3-label-4" x="912.0" y="947" font-size="9" text-anchor="middle" fill="#555555">2</text>
  <rect id="node-box-932-904" x="932" y="904" width="40" height="30" rx="0" fill="#a3e4d7" stroke="#117a65" stroke-width="1.3"/>
  <text id="node-expand-step3-label-5" x="952.0" y="947" font-size="9" text-anchor="middle" fill="#555555">3</text>
  <rect id="node-box-972-904" x="972" y="904" width="40" height="30" rx="0" fill="#f9e79f" stroke="#b7950b" stroke-width="1.3"/>
  <text id="node-expand-step3-label-6" x="992.0" y="947" font-size="9" text-anchor="middle" fill="#555555">4</text>
  <rect id="node-box-1012-904" x="1012" y="904" width="40" height="30" rx="0" fill="#f5b7b1" stroke="#922b21" stroke-width="1.3"/>
  <text id="node-expand-step3-label-7" x="1032.0" y="947" font-size="9" text-anchor="middle" fill="#555555">5</text>
  <rect id="node-box-1052-904" x="1052" y="904" width="40" height="30" rx="0" fill="#d5f5e3" stroke="#1e8449" stroke-width="2.5"/>
  <text id="node-expand-step3-label-8" x="1072.0" y="947" font-size="9" text-anchor="middle" fill="#555555">6</text>
  <rect id="node-box-1092-904" x="1092" y="904" width="40" height="30" rx="0" fill="#fdebd0" stroke="#ca6f1e" stroke-width="2.5"/>
  <text id="node-expand-step3-label-9" x="1112.0" y="947" font-size="9" text-anchor="middle" fill="#555555">7</text>
  <rect id="node-chunk-new-1" data-k="K2,K6" x="1052" y="966" width="40" height="30" rx="0" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.3"/>
  <text id="node-chunk-new-1-label-1" x="1072.0" y="985" font-size="10" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">#1</text>
  <line id="edge-expand-step3-chunk-new-1" data-k="K3,K14" x1="1072.0" y1="950" x2="1072.0" y2="966" stroke="#333333" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect id="node-chunk-new-9" data-k="K2,K6" x="1092" y="966" width="40" height="30" rx="0" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.3"/>
  <text id="node-chunk-new-9-label-1" x="1112.0" y="985" font-size="10" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ca6f1e">#9</text>
  <line id="edge-expand-step3-chunk-new-9" data-k="K3,K14" x1="1112.0" y1="950" x2="1112.0" y2="966" stroke="#333333" stroke-width="1.5" marker-end="url(#arr)"/>
  <text id="node-expand-step3-label-10" x="812" y="990" font-size="11" fill="#333333">段长 8 格，地址仍然连续</text>
  <text id="node-expand-step3-label-11" x="812" y="1006" font-size="11" fill="#333333">新 chunk（HBM）只出现在段尾</text>
  <line id="edge-free-free" data-k="K14" x1="402" y1="878" x2="418" y2="878" stroke="#225c33" stroke-width="1.5" marker-end="url(#arr)"/>
  <line id="edge-free-free-2" data-k="K14" x1="782" y1="878" x2="798" y2="878" stroke="#225c33" stroke-width="1.5" marker-end="url(#arr)"/>
  </g>
</svg>
```
