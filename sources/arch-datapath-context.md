# AMD / Hopper / Blackwell 片上存储访问对照：最终上下文

## 1. 范围与参考

这份上下文对应图 `amd_hopper_blackwell_arrow_bandwidth.png`，内容是三种架构的片上存储与计算单元之间的数据通路对照。

- AMD：CDNA（MI300X 为例），LDS + MFMA。
- NVIDIA Hopper：H100，SMEM + wgmma / mma.sync。
- NVIDIA Blackwell：B200，SMEM + TMEM + tcgen05。
- 这不是某个 kernel 的图，而是指令和存储层级的对照。画法参考 Phabricator paste P2553123029 的结构（分节、术语、数据流、偏好），但内容不同。
- 数值（容量、带宽）来自公开规格，图中的带宽是峰值理论估算，未经实测，也没有逐条对照 CDNA ISA 或 PTX 文档。

## 1.5 图要表达的核心问题

同样是“片上共享存储 + 矩阵单元”，AMD 和 NVIDIA 的数据路径不同，代价也不同。图要回答：

1. 计算单元能不能直接读片上存储？（⑤）AMD MFMA 不能读 LDS，必须先经 VGPR；Hopper wgmma 直接读 SMEM；Blackwell tcgen05.mma 直接读 SMEM。
2. 累加器放在哪里？（⑦⑧）AMD 和 Hopper 在寄存器；Blackwell 在 TMEM，需要 tcgen05.ld/st 搬运。
3. 异步完成怎么同步？（⑨）AMD 用 s_waitcnt lgkmcnt 计数；Hopper 用 mbarrier；Blackwell 用 mbarrier 加 tcgen05.commit。
4. 线程之间怎么交换？（⑩）AMD 的 DPP 在寄存器内完成，ds_bpermute 占用 LDS 带宽；NVIDIA 的 shfl.sync 不经过共享存储。
5. 为什么要先搬进片上存储？（②③④）片上带宽远高于每个计算单元分到的显存带宽（HBM 份额约 17–54 GB/s，LDS/SMEM 约 250 GB/s），数据要进片上存储被多次复用。
6. 容量差多少？（对数小条）片上是 KB 级，显存是 GB 级，相差约 10⁵ 倍，所以需要分块（tiling）。
7. 三张图的编号是否可以对照？是。①–⑩ 含义相同，虚线或 ✕ 表示该架构没有这条路径。

## 2. 术语对应

- CU（AMD）/ SM（NVIDIA）：计算单元，片上存储按它划分。
- wave（64 线程）/ warp（32 线程）：线程执行组，指令发射单位。
- workgroup（WG）/ CTA / block：共享 LDS / SMEM 的线程块。
- VGPR（AMD 向量寄存器）/ RF（NVIDIA Register File）：线程私有寄存器。
- LDS（AMD）/ SMEM（NVIDIA shared memory）：块内共享的片上存储，功能对应。
- TMEM（Blackwell 专用累加存储）：没有 AMD 或 Hopper 对应物。
- MFMA（AMD 矩阵指令）/ mma.sync、wgmma（Hopper）/ tcgen05.mma（Blackwell）。
- DPP / ds_bpermute（AMD）/ shfl.sync（NVIDIA）：线程间交换。
- s_waitcnt lgkmcnt（AMD）/ mbarrier（NVIDIA）：异步完成同步。
- TMA：Hopper 与 Blackwell 的张量搬运单元，每 SM 一个，由单线程发起。

## 3. 图的结构

三块竖排，每块组件位置一致，便于上下对照。

- 编号 ①–⑩ 在三块图中含义相同。
  - ① 全局显存 → 寄存器的读取。
  - ② 全局显存 → LDS / SMEM 的异步写入（AMD `global_load_lds`，Hopper `cp.async` / TMA，Blackwell TMA）。
  - ③ 寄存器 → 片上存储的写入。
  - ④ 片上存储 → 寄存器的读取。
  - ⑤ 片上存储 → 计算单元的直接读取（AMD 不支持，画成虚线）。
  - ⑥ 寄存器 → 计算单元的操作数读取（Blackwell 不经过寄存器，画成虚线）。
  - ⑦ TMEM ↔ 计算单元的累加（只有 Blackwell 有）。
  - ⑧ TMEM ↔ 寄存器的 `tcgen05.ld` / `st`（只有 Blackwell 有）。
  - ⑨ 异步完成同步。
  - ⑩ 线程间交换。
- 实线表示有对应操作，虚线或 ✕ 表示无对应。
- 箭头标注直接写在箭头旁边，不使用右侧的说明栏。
- 方框内写容量、关键限制，并在底部放一根对数刻度的小条。

## 4. 容量（方框内标注，对数刻度）

| 存储 | AMD (MI300X) | Hopper (H100) | Blackwell (B200) |
|---|---|---|---|
| 寄存器 | VGPR 512 KB / CU | RF 256 KB / SM | RF 256 KB / SM |
| 共享存储 | LDS 64 KB / CU（WG 私有，32 banks） | SMEM ≤228 KB / SM（单 block 上限 227 KB，与 L1 共用） | SMEM ≤228 KB / SM |
| TMEM | 无 | 无 | 256 KB / SM（128 lane × 512 列 × 32 bit） |
| 显存 HBM | 192 GB | 80 GB | 192 GB |

- 对数刻度：每格 10 倍，三块图使用同一标尺。HBM 与片上存储相差约 10⁵ 倍，线性刻度下片上存储几乎看不见。
- 小条的标尺：1 KB 到 100 GB。

## 5. 带宽（箭头旁，橙色字，峰值理论估算）

| 路径 | AMD | Hopper | Blackwell |
|---|---|---|---|
| ① 全局显存 → 寄存器 | ≈17 GB/s/CU | ≈25 GB/s/SM | ≈54 GB/s/SM |
| ② 全局显存 → LDS / SMEM | ≈17 GB/s/CU | ≈25 GB/s/SM | ≈54 GB/s/SM |
| ③④ 寄存器 ↔ 共享存储 | ≈270 GB/s | ≈250 GB/s | ≈240 GB/s |
| ⑤ 共享存储 → 计算单元 | ✕ 不支持 | ≈250 GB/s（SMEM 端口） | ≈240 GB/s（SMEM 端口） |
| ⑥ 寄存器 → 计算单元 | 未量化 | 未量化 | ✕ 不经过 RF |
| ⑦⑧ TMEM 相关 | ✕ 无 | ✕ 无 | 未公开 |
| ⑩ 线程间交换 | 未量化 | 未量化 | 未量化 |

估算方法：
- HBM：总带宽 ÷ CU/SM 数，得到平均每个单元的份额。MI300X 5.3 TB/s ÷ 304 CU；H100 SXM 3.35 TB/s ÷ 132 SM；B200 8 TB/s ÷ 约 148 SM。这是平均值，命中 L2 时实际更高。
- LDS / SMEM：128 B/clk × 约 2 GHz（AMD 约 2.1 GHz，Hopper 与 Blackwell 按约 2 GHz 取整）。
- 同步（⑨）不是数据通路，没有带宽。

风险点：B200 的 SM 数（约 148）与时钟是估算，需要核对。TMEM 的带宽没有可靠公开数值，图中标为“未公开”。

## 6. 访问路径要点

### AMD（LDS + MFMA）
- MFMA 只能从寄存器（VGPR，部分含 AGPR）取操作数，不能直接读 LDS。路径是 LDS → `ds_read` → VGPR → MFMA。
- `global_load_lds` 可以把全局显存数据直接写入 LDS，绕过 VGPR，但读 LDS 仍然要经过寄存器。
- 累加器在寄存器中，没有专用累加存储。
- LDS 与 L1 独立，按 WG 分配，容量影响每 CU 驻留的 WG 数。
- `lgkmcnt` 与标量内存、消息共用，同时在途的计数有上限（约 15）。
- wave 为 64 线程，bank 冲突按 wavefront 计算，移植时需要重新排布。

### Hopper（SMEM + wgmma）
- `wgmma` 可以直接从 SMEM 读取 A/B 操作数，不经过寄存器；累加器仍在 RF 中。
- `mma.sync` 的操作数来自 RF，需要先用 `ldmatrix` 从 SMEM 搬入。
- `cp.async` 和 TMA 异步写入 SMEM，用 `mbarrier` 等待完成。TMA 每 SM 一个单元，由单线程发起。
- `wgmma` 的发起单位是 warpgroup（4 个 warp，128 线程），异步执行，用 commit / wait_group 管理。
- SMEM 与 L1 共用容量；cluster 内的 DSMEM 可以跨 block 访问（未画在图中）。

### Blackwell（SMEM + TMEM + tcgen05）
- `tcgen05.mma` 由单线程发射，异步执行，操作数从 SMEM 读取（descriptor 指定地址），不经过 RF。
- 累加器放在 TMEM，`tcgen05.ld` / `st` 负责 TMEM 与寄存器之间的搬运。
- 一个 warp 只能访问它所在 warpgroup 中对应的 32 个 lane。
- `tcgen05.commit` 把 MMA 完成映射到 mbarrier。
- `tcgen05.cp` 可以把 SMEM 的数据拷贝到 TMEM。
- 一个 CTA 的 TMEM 列数有上限，`tcgen05.alloc` 按列分配，列数为 2 的幂。

### FA 数据流（Blackwell 示例，只画在图底部）
TMA② → SMEM → mbarrier⑨ → tcgen05.mma⑤ → TMEM⑦ → tcgen05.ld⑧ → softmax → P → tcgen05.mma → TMEM → 写回。

## 7. 渲染产物与复现

- 最终 PNG：`/tmp/lds_compare/amd_hopper_blackwell_arrow_bandwidth.png`，2000×3400（2× 渲染）。
- 源文件：`/tmp/lds_compare/amd_hopper_blackwell_arrow_bandwidth.svg`（1000×1700，见第 9 节）。
- 渲染脚本：`/tmp/lds_compare/render_png.py`（见第 10 节）。
- 命令：`/usr/bin/python3 /tmp/lds_compare/render_png.py <svg> <png> [倍率]`。

渲染注意事项：
- 本机 librsvg 版本较旧，不支持 `rsvg_handle_get_intrinsic_size_in_pixels`，脚本从 SVG 根元素读取 `width` / `height`，所以 SVG 必须写数字，不能用 `100%`。
- 不支持 `paint-order`，给文字加白色描边会把文字盖住。文字不要用描边。
- 不支持 CSS `var(--...)`，颜色必须写死。
- 中文字体使用 `Noto Sans CJK SC`，本机已安装。
- 修改 SVG 后要重新渲染，并打开 PNG 检查文字和箭头是否被遮挡。

## 8. 用户偏好（本对话中确认的）

- 直观、少文字：标注直接放在箭头旁边或方框里，不要边栏、不要底部大段说明。
- 竖排三块，组件位置一致，方便左右上下对照。
- 不删除之前已有的信息；修改时只增加或调整位置。
- 编号 ①–⑩ 在三块图中含义相同；虚线 / ✕ 表示无对应。
- 容量和带宽用可视化的小条或橙色字标注，并注明估算性质。
- 中文说明，AMD 术语可以加 NVIDIA 对应项。
- PNG 是主要产物，同时保留 SVG 源文件和渲染脚本。
- 不要重复注解；需要强调的地方用箭头或框表达。

## 9. 完整 SVG 源码

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1700" width="1000" height="1700">
<title>AMD、Hopper、Blackwell 竖排对照图，箭头标注带宽估算</title>
<desc>三个架构竖排，组件位置一致。箭头标注编号、指令和带宽估算，存储方框内标注容量与对数小条。带宽为峰值理论估算，未公开的标注为未量化。</desc>
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#6b7280"/>
  </marker>
</defs>
<style>
  .bg { fill: #ffffff; }
  .pn { fill: #f9fafb; stroke: #d1d5db; stroke-width: 1; }
  .nd { fill: #ffffff; stroke-width: 1.6; }
  .gh { fill: none; stroke: #9ca3af; stroke-width: 1.2; stroke-dasharray: 4 3; }
  .trk { fill: none; stroke: #d1d5db; stroke-width: 1; }
  .ln { stroke: #6b7280; stroke-width: 1.5; fill: none; }
  .ds { stroke-dasharray: 5 4; stroke: #9ca3af; }
  .pt { font-size: 15px; font-weight: 700; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .bt { font-size: 13px; font-weight: 700; fill: #111827; text-anchor: middle; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .s { font-size: 10.5px; fill: #4b5563; text-anchor: middle; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .lb { font-size: 11.5px; fill: #111827; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .bw { font-size: 11px; font-weight: 700; fill: #b45309; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .key { font-size: 12.5px; font-weight: 700; fill: #111827; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .legend { font-size: 12px; fill: #4b5563; text-anchor: middle; font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; }
</style>

<rect class="bg" x="0" y="0" width="1000" height="1700"/>

<!-- ============ AMD CDNA ============ -->
<g transform="translate(0,0)">
  <rect class="pn" x="0" y="0" width="1000" height="510" rx="12"/>
  <text class="pt" x="24" y="30" style="fill:#ED1C24">AMD CDNA：LDS + MFMA</text>

  <path class="ln" d="M450 110 L200 200" marker-end="url(#ah)"/>
  <text class="lb" x="318" y="142" text-anchor="end">① global_load 读入 VGPR</text>
  <text class="bw" x="318" y="156" text-anchor="end">≈17 GB/s/CU</text>
  <path class="ln" d="M520 110 L520 200" marker-end="url(#ah)"/>
  <text class="lb" x="530" y="140">② global_load_lds 直写 LDS</text>
  <text class="bw" x="530" y="154">≈17 GB/s/CU</text>
  <path class="ln" d="M270 222 L385 222" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="206" text-anchor="middle">③ ds_write</text>
  <text class="bw" x="327" y="218" text-anchor="middle">≈270 GB/s</text>
  <path class="ln" d="M385 262 L270 262" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="274" text-anchor="middle">④ ds_read</text>
  <text class="bw" x="327" y="288" text-anchor="middle">≈270 GB/s</text>
  <path class="ln" d="M615 215 L790 110" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="745" y="150">⑨ s_waitcnt lgkmcnt</text>
  <path class="ln" d="M190 290 L190 390" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="200" y="380">⑩ DPP · 未量化</text>
  <path class="ln" d="M240 290 L385 425" marker-end="url(#ah)"/>
  <text class="lb" x="300" y="330">⑥ 读 VGPR · 未量化</text>
  <path class="ln ds" d="M500 290 L500 390"/>
  <text class="lb" x="512" y="316">⑤ ✕ MFMA 不能直读 LDS</text>
  <text class="s" x="512" y="332" style="text-anchor:start">无带宽（路径不存在）</text>

  <rect class="nd" x="385" y="40" width="230" height="70" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="500" y="62">全局显存 Global</text>
  <text class="s" x="500" y="86">HBM · 192 GB</text>
  <rect class="trk" x="400" y="98" width="200" height="6"/>
  <rect x="400" y="98" width="195" height="6" fill="#ED1C24"/>

  <rect class="nd" x="760" y="40" width="200" height="70" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="860" y="62">⑨ 同步</text>
  <text class="s" x="860" y="94">s_waitcnt lgkmcnt(0)</text>
  <text class="s" x="860" y="108">计数器与标量内存共用，约 15 条上限</text>

  <rect class="nd" x="40" y="200" width="230" height="90" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="155" y="222">VGPR · 512 KB / CU</text>
  <text class="s" x="155" y="240">向量寄存器 · wave64（64 线程）</text>
  <text class="s" x="155" y="254">MFMA 操作数只能来自 VGPR / AGPR</text>
  <rect class="trk" x="55" y="280" width="200" height="6"/>
  <rect x="55" y="280" width="64" height="6" fill="#ED1C24"/>

  <rect class="nd" x="385" y="200" width="230" height="90" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="500" y="222">LDS · 64 KB / CU</text>
  <text class="s" x="500" y="240">WG 私有，不可跨 WG 访问</text>
  <text class="s" x="500" y="254">约 128B/clk · 4 个 SIMD 争用</text>
  <text class="s" x="500" y="268">32 banks</text>
  <rect class="trk" x="400" y="280" width="200" height="6"/>
  <rect x="400" y="280" width="43" height="6" fill="#ED1C24"/>

  <rect class="gh" x="730" y="200" width="230" height="90" rx="8"/>
  <text class="bt" x="845" y="222">⑦⑧ ✕ 无对应</text>
  <text class="s" x="845" y="240">无专用累加存储</text>
  <text class="s" x="845" y="254">累加器在 VGPR / AGPR</text>

  <rect class="nd" x="40" y="390" width="230" height="64" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="155" y="412">⑩ 线程间交换</text>
  <text class="s" x="155" y="430">DPP 寄存器内 · ds_bpermute 占 LDS</text>

  <rect class="nd" x="385" y="390" width="230" height="70" rx="8" style="stroke:#ED1C24"/>
  <text class="bt" x="500" y="412">MFMA</text>
  <text class="s" x="500" y="430">v_mfma · 每 CU 约 4 个 SIMD 并行</text>

  <text class="key" x="24" y="470">关键：LDS 不能直接喂给 MFMA，必须先经 VGPR</text>
</g>

<!-- ============ NVIDIA Hopper ============ -->
<g transform="translate(0,540)">
  <rect class="pn" x="0" y="0" width="1000" height="510" rx="12"/>
  <text class="pt" x="24" y="30" style="fill:#3d7000">NVIDIA Hopper：SMEM + wgmma</text>

  <path class="ln" d="M450 110 L200 200" marker-end="url(#ah)"/>
  <text class="lb" x="318" y="142" text-anchor="end">① ld.global 读入 RF</text>
  <text class="bw" x="318" y="156" text-anchor="end">≈25 GB/s/SM</text>
  <path class="ln" d="M520 110 L520 200" marker-end="url(#ah)"/>
  <text class="lb" x="530" y="140">② cp.async / TMA 异步写 SMEM</text>
  <text class="bw" x="530" y="154">≈25 GB/s/SM</text>
  <path class="ln" d="M270 222 L385 222" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="206" text-anchor="middle">③ st.shared</text>
  <text class="bw" x="327" y="218" text-anchor="middle">≈250 GB/s</text>
  <path class="ln" d="M385 262 L270 262" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="274" text-anchor="middle">④ ldmatrix</text>
  <text class="bw" x="327" y="288" text-anchor="middle">≈250 GB/s</text>
  <path class="ln" d="M615 215 L790 110" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="745" y="150">⑨ mbarrier 等待</text>
  <path class="ln" d="M190 290 L190 390" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="200" y="380">⑩ shfl · 未量化</text>
  <path class="ln" d="M240 290 L385 425" marker-end="url(#ah)"/>
  <text class="lb" x="300" y="330">⑥ mma.sync 读 RF · 未量化</text>
  <path class="ln" d="M500 290 L500 390" marker-end="url(#ah)"/>
  <text class="lb" x="512" y="316">⑤ wgmma 直读 SMEM</text>
  <text class="bw" x="512" y="332">≈250 GB/s（SMEM 端口）</text>

  <rect class="nd" x="385" y="40" width="230" height="70" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="500" y="62">全局显存 Global</text>
  <text class="s" x="500" y="86">HBM · 80 GB</text>
  <rect class="trk" x="400" y="98" width="200" height="6"/>
  <rect x="400" y="98" width="186" height="6" fill="#76B900"/>

  <rect class="nd" x="760" y="40" width="200" height="70" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="860" y="62">⑨ 同步</text>
  <text class="s" x="860" y="94">mbarrier：计数到达后唤醒</text>
  <text class="s" x="860" y="108">TMA 用字节计数完成</text>

  <rect class="nd" x="40" y="200" width="230" height="90" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="155" y="222">寄存器 RF · 256 KB / SM</text>
  <text class="s" x="155" y="240">Register File · warp 32 线程</text>
  <text class="s" x="155" y="254">mma.sync 与 wgmma 累加在 RF</text>
  <rect class="trk" x="55" y="280" width="200" height="6"/>
  <rect x="55" y="280" width="57" height="6" fill="#76B900"/>

  <rect class="nd" x="385" y="200" width="230" height="90" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="500" y="222">SMEM · ≤228 KB / SM</text>
  <text class="s" x="500" y="240">单 block 上限 227KB，需 opt-in</text>
  <text class="s" x="500" y="254">与 L1 共用 · 同地址广播</text>
  <text class="s" x="500" y="268">32 banks</text>
  <rect class="trk" x="400" y="280" width="200" height="6"/>
  <rect x="400" y="280" width="55" height="6" fill="#76B900"/>

  <rect class="gh" x="730" y="200" width="230" height="90" rx="8"/>
  <text class="bt" x="845" y="222">⑦⑧ ✕ 无 TMEM</text>
  <text class="s" x="845" y="240">累加器在寄存器中</text>
  <text class="s" x="845" y="254">wgmma 操作数可直接来自 SMEM</text>

  <rect class="nd" x="40" y="390" width="230" height="64" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="155" y="412">⑩ 线程间交换</text>
  <text class="s" x="155" y="430">shfl.sync：warp 内 32 线程</text>
  <text class="s" x="155" y="444">不经过 SMEM</text>

  <rect class="nd" x="385" y="390" width="230" height="70" rx="8" style="stroke:#76B900"/>
  <text class="bt" x="500" y="412">Tensor Core</text>
  <text class="s" x="500" y="430">wgmma：warpgroup 128 线程，异步</text>
  <text class="s" x="500" y="444">M=64，N≤256</text>

  <text class="key" x="24" y="470">关键：wgmma 可直接读 SMEM，但累加器仍在寄存器</text>
</g>

<!-- ============ NVIDIA Blackwell ============ -->
<g transform="translate(0,1080)">
  <rect class="pn" x="0" y="0" width="1000" height="510" rx="12"/>
  <text class="pt" x="24" y="30" style="fill:#2f6f00">NVIDIA Blackwell：SMEM + TMEM + tcgen05</text>

  <path class="ln" d="M450 110 L200 200" marker-end="url(#ah)"/>
  <text class="lb" x="318" y="142" text-anchor="end">① ld.global 读入 RF</text>
  <text class="bw" x="318" y="156" text-anchor="end">≈54 GB/s/SM</text>
  <path class="ln" d="M520 110 L520 200" marker-end="url(#ah)"/>
  <text class="lb" x="530" y="140">② TMA 异步写 SMEM</text>
  <text class="bw" x="530" y="154">≈54 GB/s/SM</text>
  <path class="ln" d="M270 222 L385 222" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="206" text-anchor="middle">③ st.shared</text>
  <text class="bw" x="327" y="218" text-anchor="middle">≈240 GB/s</text>
  <path class="ln" d="M385 262 L270 262" marker-end="url(#ah)"/>
  <text class="lb" x="327" y="274" text-anchor="middle">④ ld.shared</text>
  <text class="bw" x="327" y="288" text-anchor="middle">≈240 GB/s</text>
  <path class="ln" d="M615 215 L790 110" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="745" y="150">⑨ mbarrier · commit</text>
  <path class="ln" d="M190 290 L190 390" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="200" y="380">⑩ shfl · 未量化</text>
  <path class="ln ds" d="M240 290 L385 425"/>
  <text class="lb" x="300" y="330">⑥ ✕ 操作数不经 RF</text>
  <path class="ln" d="M500 290 L500 390" marker-end="url(#ah)"/>
  <text class="lb" x="512" y="316">⑤ tcgen05.mma 直读 SMEM</text>
  <text class="bw" x="512" y="332">≈240 GB/s（SMEM 端口）</text>
  <path class="ln" d="M760 290 L615 415" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="640" y="300">⑦ 累加 · 未公开</text>
  <path class="ln" d="M845 290 V345 H120 V290" marker-start="url(#ah)" marker-end="url(#ah)"/>
  <text class="lb" x="720" y="362">⑧ tcgen05.ld / st · 未公开</text>

  <rect class="nd" x="385" y="40" width="230" height="70" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="500" y="62">全局显存 Global</text>
  <text class="s" x="500" y="86">HBM · 192 GB</text>
  <rect class="trk" x="400" y="98" width="200" height="6"/>
  <rect x="400" y="98" width="195" height="6" fill="#2f6f00"/>

  <rect class="nd" x="760" y="40" width="200" height="70" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="860" y="62">⑨ 异步完成同步</text>
  <text class="s" x="860" y="94">mbarrier · tcgen05.commit</text>
  <text class="s" x="860" y="108">MMA 完成后 arrive 唤醒</text>

  <rect class="nd" x="40" y="200" width="230" height="90" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="155" y="222">寄存器 RF · 256 KB / SM</text>
  <text class="s" x="155" y="240">Register File · warp 32 线程</text>
  <text class="s" x="155" y="254">不作为 MMA 操作数（⑥ ✕）</text>
  <rect class="trk" x="55" y="280" width="200" height="6"/>
  <rect x="55" y="280" width="57" height="6" fill="#2f6f00"/>

  <rect class="nd" x="385" y="200" width="230" height="90" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="500" y="222">SMEM · ≤228 KB / SM</text>
  <text class="s" x="500" y="240">TMA 写入 · MMA 经 descriptor 读取</text>
  <text class="s" x="500" y="254">每 SM 总量随 L1 carveout 变化</text>
  <rect class="trk" x="400" y="280" width="200" height="6"/>
  <rect x="400" y="280" width="55" height="6" fill="#2f6f00"/>

  <rect class="nd" x="730" y="200" width="230" height="90" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="845" y="222">⑦ TMEM · 256 KB / SM</text>
  <text class="s" x="845" y="240">128 lane × 512 列 × 32bit</text>
  <text class="s" x="845" y="254">warp 只访问对应 32 lane，按列 alloc</text>
  <rect class="trk" x="745" y="280" width="200" height="6"/>
  <rect x="745" y="280" width="57" height="6" fill="#2f6f00"/>

  <rect class="nd" x="40" y="390" width="230" height="64" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="155" y="412">⑩ 线程间交换</text>
  <text class="s" x="155" y="430">shfl.sync · 不经 TMEM / SMEM</text>

  <rect class="nd" x="385" y="390" width="230" height="70" rx="8" style="stroke:#2f6f00"/>
  <text class="bt" x="500" y="412">Tensor Core</text>
  <text class="s" x="500" y="430">tcgen05.mma：单线程发射，异步</text>
  <text class="s" x="500" y="444">M=64/128（2-CTA 时 256）· N≤256</text>

  <text class="key" x="24" y="470">关键：累加器移到 TMEM，操作数仍从 SMEM 读取</text>
  <text class="s" x="24" y="488" style="text-anchor:start">FA 流：TMA② → SMEM → mbarrier⑨ → tcgen05.mma⑤ → TMEM⑦ → tcgen05.ld⑧ → softmax → P → tcgen05.mma → TMEM → 写回</text>
</g>

<text class="legend" x="500" y="1645">实线 = 有对应　　虚线 / ✕ = 无对应　　编号相同 = 同一功能</text>
<text class="legend" x="500" y="1666">方框内小条 = 容量，对数刻度（每格 10 倍），三图统一标尺</text>
<text class="legend" x="500" y="1687">橙色为带宽估算：HBM = 总带宽 ÷ CU/SM 数；LDS/SMEM ≈ 128 B/clk × 约 2 GHz；未公开的标注为未量化</text>
</svg>
```

## 10. 渲染脚本源码

```python
import ctypes as c
import sys
from pathlib import Path

# Usage: /usr/bin/python3 render_png.py <input.svg> <output.png> [scale]
src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

R = c.CDLL('librsvg-2.so.2')
C = c.CDLL('libcairo.so.2')
G = c.CDLL('libgobject-2.0.so.0')

R.rsvg_handle_new_from_file.argtypes = [c.c_char_p, c.POINTER(c.c_void_p)]
R.rsvg_handle_new_from_file.restype = c.c_void_p


class Rect(c.Structure):
    _fields_ = [('x', c.c_double), ('y', c.c_double), ('width', c.c_double), ('height', c.c_double)]


R.rsvg_handle_render_document.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(Rect), c.POINTER(c.c_void_p)]
R.rsvg_handle_render_document.restype = c.c_int
C.cairo_image_surface_create.argtypes = [c.c_int, c.c_int, c.c_int]
C.cairo_image_surface_create.restype = c.c_void_p
C.cairo_create.argtypes = [c.c_void_p]
C.cairo_create.restype = c.c_void_p
C.cairo_scale.argtypes = [c.c_void_p, c.c_double, c.c_double]
C.cairo_surface_write_to_png.argtypes = [c.c_void_p, c.c_char_p]
C.cairo_surface_write_to_png.restype = c.c_int
C.cairo_destroy.argtypes = [c.c_void_p]
C.cairo_surface_destroy.argtypes = [c.c_void_p]
G.g_object_unref.argtypes = [c.c_void_p]

err = c.c_void_p()
handle = R.rsvg_handle_new_from_file(str(src).encode(), c.byref(err))
assert handle and not err.value, 'failed to load SVG'

import xml.etree.ElementTree as ET
root = ET.parse(src).getroot()
W, H = int(float(root.attrib['width'])), int(float(root.attrib['height']))

surface = C.cairo_image_surface_create(0, int(W * scale), int(H * scale))
ctx = C.cairo_create(surface)
C.cairo_scale(ctx, scale, scale)
assert R.rsvg_handle_render_document(handle, ctx, c.byref(Rect(0, 0, W, H)), c.byref(err))
assert C.cairo_surface_write_to_png(surface, str(dst).encode()) == 0, 'write failed'
C.cairo_destroy(ctx)
C.cairo_surface_destroy(surface)
G.g_object_unref(handle)
print('Rendered PNG', int(W * scale), int(H * scale), '->', dst)
```
