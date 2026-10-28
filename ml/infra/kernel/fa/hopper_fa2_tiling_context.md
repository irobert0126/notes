# hopper_fa2 tiling：最终知识点、完整 SVG 与 PNG renderer

PNG分享链接：https://pxl.cl/gq9zb

最新本地SVG：`/tmp/fa4_learn/hopper_fa2_tiling_section2_summary.svg`

最新本地PNG：`/tmp/fa4_learn/hopper_fa2_tiling_section2_summary.png`

## FA2 / Hopper：此图配置与最终知识点

- 这是标准 FA2 CUDA 的传统 `mma.sync` / `cp.async` 路径，不是 FA3 Hopper 专用实现。
- BF16、D128、noncausal、dropout=0：B_M=128，B_N=64，4warps/128threads；Is_Q_in_regs=false，Share_Q_K_smem=false。
- 示例 L=6×64=384，因此 Q 有3个128行块。一个 CTA 处理一个逻辑固定 Q block。
- TiledMMA 为 SM80_16x8x16 BF16/F32，warp在M方向布局4×1×1，Q行分片为交错16行段：w0=0–15与64–79，w1=16–31与80–95，w2=32–47与96–111，w3=48–63与112–127。不能画成 w0连续前32行。
- 每个warp的lane继续细分矩阵片段；搬运所有权由copy布局决定，与MMA计算所有权不同，不是每warp各搬一个完整K/V副本。
- 同一批warps搬运并完成QK、softmax/O修正、PV；cp.async V[j]与当前QK重叠，K[j]被QK读完后可预取K[j−1]，与后续softmax/PV重叠。
- Q先入SMEM；此配置不会将整块Q永久保持在寄存器。mma.sync操作数通过线程寄存器提供；S/P/O及在线统计分散在线程寄存器。
- SMEM中Q、K、V是独立区域；K/V各一个物理缓冲。K覆盖前确认QK读完，V覆盖前确认PV读完；不是三份KV缓冲或按每warp分配一组slot。
- 最后在线归一化O：线程寄存器 → SMEM输出staging → HBM。
- 图只画warp0/1的详细执行与两个示例lane，w2/3依相同机制处理自己的Q行。

### 配置来源

`fbcode/ai_codesign/gen_ai/flash_attention_v2/csrc/flash_attn/src/`：
- `flash_fwd_launch_template.h`：D128 Hopper启动选择 B_M128/B_N64/4warps。
- `kernel_traits.h`：TiledMMA、Is_Q_in_regs、Share_Q_K_smem与存储区域。
- `flash_fwd_kernel.h`：Q/K/V存储、倒序KV扫描、V与QK重叠、下一K预取、输出。

## 当前图的范围与读法

这是 tiling 图的最终三点摘要版，**不是另一张 equal_op_work pipeline 比较图**。此图显示真实所选 kernel 的 tile 配置、角色、lane 分片和存储路径；不能根据等宽轮次列推断实际周期或跨架构运行速度。

所选配置是 BF16、D=128、noncausal forward。各图有6个 KV block 的例子，但因为 B_N 不同，例子的 L 也不同；不能将不同图的 Q tile 数直接当作实现优劣。

### 矩阵与维度

- B=batch，H=heads，L=sequence，D=head dimension。先选一组(b,h)，Q/K/V 的逻辑矩阵是 L×D；实际张量 stride/layout 不必与逻辑图相同。
- B_M 沿 Q 的 query 行切块；B_N 沿 K/V 序列切块。固定 Q[i] 为 B_M×D，每轮读 K[j]、V[j] 各 B_N×D。
- Kᵀ 是 D×L，因此图中 Kᵀ 竖着按列切，V 横着按行切。不是把 K 的 head dim D 切成 KV block。
- S[j] = Q[i] K[j]ᵀ，尺寸 B_M×B_N；P[j] 是在线 softmax 的概率片段；O[i] 是 B_M×D 的累加输出。
- 图中 S/P 的 L×L 背景是逻辑关系，不表示把整张 attention score/probability 写进 HBM。
- 在本 noncausal 配置中，固定 Q 的任务遍历 ceil(L/B_N) 个 KV tile；不能误写成 L/B_M。
- 图示 j=5→0 按所选实现倒序遍历，固定 Q[i] 每轮复用。倒序是 kernel 实现选择，不是 NVIDIA 硬件要求。
- O 每轮更新在线累积；最后才归一化并写回。图省略 LSE 和部分辅助统计。

### 四个 section

1. Block tile：原始 Q/K/V 的切块、固定 Q 行块、S 内的 j 循环箭头、O 累积。
2. CTA 协作：2a 固定 Q 沿 KV 循环；2b 一轮的 Q/S/O 行分片并复用完整 K/V；2c 角色执行流、启动请求、预取及物理缓冲内容。右侧只有三点：固定 Q 与 KV 遍历、warp 分工、缓冲复用。
3. Lane tiling：片段进一步分散到线程私有元素；只画两条示例 lane。warp 是32线程执行组，lane 对应线程；矩阵指令与 Tensor Core 执行单元不能与 lane 一一配对。
4. 存储：从 HBM、SMEM 到寄存器/TMEM，区分 block 共享 tile 和 lane 私有片段，及最终 O 写回。

### 异步执行和同步

- 发起读取/搬运 = worker 发出异步请求；真实传输由硬件推进，不表示发请求的 warp 全程等待或搬运。
- producer 等目标 slot 空闲才能写；consumer 等数据就绪才能读；消费完成并释放之后才能覆盖。
- slot 只能保证安全复用，不能保证搬运一定及时；带宽、队列、资源与数据依赖仍可能产生等待。
- 2c 的列按当前 KV 计算轮次对齐，框内写清请求的是当前块还是下一块；实际 producer 可领先当前计算。
- 物理槽表是合法占用/到齐快照，不是精确硬件周期模拟。将请求和计算画在同一列，不表示所有动作同时开始或结束。
- 不同 worker 复用完整 K/V，但计算自己的 Q/S/O 行；同名 S 必须有 worker/行归属，不能误读成重复做整个 softmax。

### Terminology

- CTA / thread block：与 AMD workgroup 的功能类比，是协作线程集合。
- warp：NVIDIA32线程执行组；AMD 对应 wave（此讨论中64线程）。不是 Tensor Core/MFMA 硬件执行单元。
- SMEM：CTA共享存储；AMD 对应 LDS。寄存器则属于各线程/lane 的私有逻辑状态。
- mma.sync / WGMMA / tcgen05：矩阵指令；Tensor Core 是执行它们的硬件。AMD MFMA 是矩阵指令的功能类比。
- warpgroup = 4个连续warp、128线程。FA4仍可把4个warp组织成组；关键变化是矩阵指令发起范围与角色职责。
- TMA 是硬件搬运机制；TMEM 是 Blackwell 的矩阵中间存储。不能将 TMA、TMEM、warp 都当作计算线程。

## 绘图表达规则（用户反复确认）

- 沿用 AMD 图的四层布局；B蓝、H青绿、L橙、D紫彩色 box 在顶部/参数区/底部保持一致。
- 矩阵名称和整块尺寸放在 box 外；内部分片只标行范围与归属。Q 与 S 的对应行保持对齐。
- S 的 j 循环箭头画在固定 Q 的行块内；Kᵀ按列、V按行切。
- 启动请求独立画；K/V 的预取可拆为同一个 worker 的两条操作行。不会因此多出一个 worker。
- 等待紧贴具体动作；不要把等待拉成整个 iteration，也不要把空白当成精确等待时长。
- 用真实 box 展示物理 slot、当前轮内容和消费后的覆盖复用。
- 左侧减少重复段落，保留数据归属、尺寸、动作、slot；解释收进右侧三点。
- 保留完整 SVG 与 PNG；每次修改渲染并检查文字截断、重叠、下标、搬运与 slot 是否对应。不要恢复旧版本排版。

## 跨版本理解：FA3 WG 与 FA4 softmax组

FA3的两个consumer WG按Q行分工，各自做完整QK/softmax/PV。FA4的两个softmax组只做各自S行的softmax，矩阵由w12发起、O由correction组修正；它们的职责更窄。FA4仍可将连续4warp组织为一组，区别是硬件矩阵指令的参与范围与软件角色。

不能根据warp/group数、不同例子中的Q块数，或stage数量判断谁一定更快。此paste不提供跨GPU性能结论。

## 复现与后续修改

直接将下方完整SVG另存为图文件，将renderer另存为`render_svg.py`：

```bash
/usr/bin/python3 render_svg.py INPUT.svg OUTPUT.png
```

SVG1280×3100，PNG2560×6200。renderer以Python ctypes调用系统librsvg/Cairo，不需要PIL、cairosvg或pip安装；需要中文字体及`librsvg-2.so.2`、`libcairo.so.2`、`libgobject-2.0.so.0`。

若修改图，使用下方生成器并调整ROOT输出目录。为使本paste独立复现，该版本生成器仅运行当前图配置，但保留共享绘图分支。源自`gen_nvidia_aligned.py`；原本`gen_nvidia_tiling.py`是调用它的wrapper。

- `hopper_fa2_tiling_section2_summary.svg` SHA256：`70d1484b5cf2643026ff89c4ecc55ab63d3931de19d1c8dde368947bda74a4dc`
- `hopper_fa2_tiling_section2_summary.png` SHA256：`d95e2a1517b67288acb7a0054ca95d392f5e819fe3515512061c87cbcbcb420d`


## 完整SVG

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="3100" viewBox="0 0 1280 3100" font-family="Noto Sans CJK SC,Arial,sans-serif" role="img">
<title>Hopper FA2 tiling：与 AMD 图相同的四层展开</title><desc>Block tile，CTA 内分片与执行流，lane 元素和矩阵执行，以及完整存储层级。配置为 BF16 D128 noncausal forward。</desc>
<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#64748b"/></marker></defs>
<rect x="0" y="0" width="1280" height="3100" rx="5" fill="#ffffff" stroke="#ffffff"/>
<text x="40" y="40" font-size="25" fill="#0f172a" font-weight="700">Hopper FA2 tiling：B / L / H / D 落在 kernel 的哪一层</text>
<text x="40" y="68" font-size="12" fill="#334155">BF16 · D=128 · noncausal forward · BM=128 · BN=64 · 4 warps · 128 threads</text>
<text x="40" y="91" font-size="11" fill="#334155">FA2：标准 CUDA、dropout=0；FA3：dense / no split / no paged KV；FA4：m_block=128、Q stage=2。</text>
<rect x="40" y="98" width="26" height="15" rx="5" fill="#2563eb" stroke="#2563eb"/>
<text x="53.0" y="109.5" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">B</text>
<text x="74" y="110" font-size="10" fill="#334155">batch</text>
<rect x="195" y="98" width="26" height="15" rx="5" fill="#0d9488" stroke="#0d9488"/>
<text x="208.0" y="109.5" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">H</text>
<text x="229" y="110" font-size="10" fill="#334155">heads</text>
<rect x="350" y="98" width="26" height="15" rx="5" fill="#ea580c" stroke="#ea580c"/>
<text x="363.0" y="109.5" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">L</text>
<text x="384" y="110" font-size="10" fill="#334155">sequence</text>
<rect x="535" y="98" width="26" height="15" rx="5" fill="#7c3aed" stroke="#7c3aed"/>
<text x="548.0" y="109.5" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">D</text>
<text x="569" y="110" font-size="10" fill="#334155">head dim</text>
<rect x="30" y="118" width="880" height="390" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="145" font-size="16" fill="#0f172a" font-weight="700">① Block tile：原始 Q/K/V → Q 行块固定，沿 K/V 序列切块</text>
<text x="48" y="168" font-size="11" fill="#334155">一次 CTA 工作：Q[i] 128×D；K[j]/V[j] 64×D；S/P 128×64；O[i] 128×D。</text>
<text x="48" y="190" font-size="10" fill="#334155">一组 (b,h)，示例 L=6×BN=384；橙框固定 Q 的第 i 块，橙色箭头表示 KV 循环 j。</text>
<text x="96.0" y="218" font-size="20" fill="#2563eb" font-weight="700" text-anchor="middle">Q</text>
<rect x="76" y="231" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="73" y="234" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="70" y="237" width="52" height="158" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<rect x="70" y="237.0" width="52" height="52.666666666666664" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<rect x="70" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<rect x="70" y="342.3333333333333" width="52" height="52.666666666666664" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<rect x="70" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="96.0" y="319.85" font-size="11" fill="#334155" text-anchor="middle">i</text>
<rect x="70" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="none" stroke="#ea580c"/>
<text x="96.0" y="417" font-size="11" fill="#334155" text-anchor="middle">L × D</text>
<text x="269.0" y="218" font-size="20" fill="#16a34a" font-weight="700" text-anchor="middle">Kᵀ</text>
<rect x="191" y="296" width="168" height="55" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="188" y="299" width="168" height="55" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="185" y="302" width="168" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="185.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="213.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="241.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="269.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="297.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="325.0" y="302" width="28.0" height="55" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<rect x="213.0" y="302" width="28.0" height="55" rx="5" fill="#bbf7d0" stroke="#ea580c"/>
<text x="227.0" y="333.35" font-size="11" fill="#334155" text-anchor="middle">j</text>
<text x="269.0" y="417" font-size="11" fill="#334155" text-anchor="middle">D × L</text>
<text x="515.0" y="218" font-size="20" fill="#dc2626" font-weight="700" text-anchor="middle">S / P</text>
<rect x="431" y="231" width="180" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="428" y="234" width="180" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="425" y="237" width="180" height="158" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<rect x="425" y="237.0" width="180" height="52.666666666666664" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<rect x="425" y="289.6666666666667" width="180" height="52.666666666666664" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<rect x="425" y="342.3333333333333" width="180" height="52.666666666666664" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<line x1="455.0" y1="237" x2="455.0" y2="395" stroke="#dc2626" stroke-dasharray="3,3"/>
<line x1="485.0" y1="237" x2="485.0" y2="395" stroke="#dc2626" stroke-dasharray="3,3"/>
<line x1="515.0" y1="237" x2="515.0" y2="395" stroke="#dc2626" stroke-dasharray="3,3"/>
<line x1="545.0" y1="237" x2="545.0" y2="395" stroke="#dc2626" stroke-dasharray="3,3"/>
<line x1="575.0" y1="237" x2="575.0" y2="395" stroke="#dc2626" stroke-dasharray="3,3"/>
<rect x="425" y="289.6666666666667" width="180" height="52.666666666666664" rx="5" fill="none" stroke="#ea580c"/>
<rect x="455.0" y="289.6666666666667" width="30.0" height="52.666666666666664" rx="5" fill="#fca5a5" stroke="#ea580c"/>
<text x="470.0" y="308.1" font-size="8" fill="#334155" text-anchor="middle">j</text>
<line x1="597" y1="274.92" x2="433" y2="274.92" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="595" y="255.43333333333334" font-size="8" fill="#94a3b8" text-anchor="end">i=0 · CTA work</text>
<line x1="597" y1="327.5866666666667" x2="433" y2="327.5866666666667" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="595" y="308.1" font-size="8" fill="#ea580c" font-weight="700" text-anchor="end">i=1 · 循环 j</text>
<line x1="597" y1="380.25333333333333" x2="433" y2="380.25333333333333" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="595" y="360.76666666666665" font-size="8" fill="#94a3b8" text-anchor="end">i=2 · CTA work</text>
<text x="515.0" y="417" font-size="11" fill="#334155" text-anchor="middle">L × L（不整块存储）</text>
<text x="726.0" y="218" font-size="20" fill="#7c3aed" font-weight="700" text-anchor="middle">V</text>
<rect x="706" y="231" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="703" y="234" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="700" y="237" width="52" height="158" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="237.0" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="263.3333333333333" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="289.6666666666667" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="316.0" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="342.3333333333333" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="368.66666666666663" width="52" height="26.333333333333332" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<rect x="700" y="263.3333333333333" width="52" height="26.333333333333332" rx="5" fill="#ddd6fe" stroke="#ea580c"/>
<text x="726.0" y="280.35" font-size="11" fill="#334155" text-anchor="middle">j</text>
<text x="726.0" y="417" font-size="11" fill="#334155" text-anchor="middle">L × D</text>
<text x="846.0" y="218" font-size="20" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<rect x="826" y="231" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="823" y="234" width="52" height="158" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="820" y="237" width="52" height="158" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<rect x="820" y="237.0" width="52" height="52.666666666666664" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<rect x="820" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<rect x="820" y="342.3333333333333" width="52" height="52.666666666666664" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<rect x="820" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="846.0" y="319.85" font-size="11" fill="#334155" text-anchor="middle">i</text>
<rect x="820" y="289.6666666666667" width="52" height="52.666666666666664" rx="5" fill="none" stroke="#ea580c"/>
<text x="846.0" y="417" font-size="11" fill="#334155" text-anchor="middle">L × D</text>
<text x="150" y="337" font-size="22" fill="#334155" text-anchor="middle">×</text>
<text x="385" y="337" font-size="22" fill="#334155" text-anchor="middle">=</text>
<text x="644" y="274" font-size="12" fill="#b45309" font-weight="700" text-anchor="middle">S → P</text>
<text x="644" y="305" font-size="10" fill="#334155" text-anchor="middle">softmax</text>
<text x="661" y="337" font-size="22" fill="#334155" text-anchor="middle">×</text>
<text x="802" y="337" font-size="22" fill="#334155" text-anchor="middle">=</text>
<line x1="353" y1="280" x2="185" y2="280" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="269" y="269" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">循环 j：5 → 0（沿 L）</text>
<text x="269" y="384" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K[j]ᵀ：D × BN</text>
<text x="515" y="443" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">橙行：固定 Q[i]，行内循环 j</text>
<line x1="770" y1="394" x2="770" y2="237" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="783" y="303" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle" transform="rotate(-90,783,303)">循环 j：5 → 0</text>
<text x="96" y="443" font-size="10" fill="#2563eb" font-weight="700" text-anchor="middle">固定 Q[i]</text>
<text x="846" y="443" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle">累积 O[i]</text>
<path d="M875,292.6666666666667 L896,292.6666666666667 L896,339.33333333333337 L876,339.33333333333337" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="48" y="467" font-size="11" fill="#334155">第 j 轮：S=Q[i]K[j]ᵀ×scale → 指数权重 P → 修正已有 O 并累加 PV[j]。</text>
<text x="48" y="492" font-size="11" fill="#334155">一个 CTA / Q tile</text>
<rect x="925" y="118" width="325" height="390" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="943" y="145" font-size="16" fill="#0f172a" font-weight="700">B / H / L / D 决定什么</text>
<rect x="940" y="164" width="26" height="18" rx="5" fill="#2563eb" stroke="#2563eb"/>
<text x="953.0" y="177.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">B</text>
<text x="970" y="178" font-size="11" fill="#334155">独立 batch：工作数 × B</text>
<rect x="940" y="205" width="26" height="18" rx="5" fill="#0d9488" stroke="#0d9488"/>
<text x="953.0" y="218.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">H</text>
<text x="970" y="219" font-size="11" fill="#334155">独立 head：工作数 × H</text>
<rect x="940" y="246" width="26" height="18" rx="5" fill="#ea580c" stroke="#ea580c"/>
<text x="953.0" y="259.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">L</text>
<text x="970" y="260" font-size="11" fill="#334155">Q 行块数 ceil(L/128)</text>
<text x="970" y="286" font-size="11" fill="#334155">KV 循环数 ceil(L/64)</text>
<rect x="940" y="316" width="26" height="18" rx="5" fill="#7c3aed" stroke="#7c3aed"/>
<text x="953.0" y="329.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">D</text>
<text x="970" y="330" font-size="11" fill="#334155">Q/K 的归约维；V/O 的列维</text>
<text x="940" y="370" font-size="11" fill="#334155" font-weight="700">grid：B × H × ceil(L / 128)</text>
<text x="940" y="406" font-size="11" fill="#334155">SM：调度 CTA 的计算资源</text>
<text x="940" y="438" font-size="11" fill="#334155">warp：32 lane；lane 是线程</text>
<text x="940" y="470" font-size="11" fill="#334155">Tensor Core：矩阵执行硬件</text>
<rect x="30" y="512" width="880" height="940" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="539" font-size="17" fill="#0f172a" font-weight="700">② CTA 内：先画分片，再画执行流与共享缓冲</text>
<text x="48" y="560" font-size="11" fill="#334155">示例：6 个 KV block，KV5 → KV0</text>
<text x="60" y="586" font-size="13" fill="#0f172a" font-weight="700">②a 固定 Q，遍历 KV</text>
<rect x="60" y="622" width="80" height="96" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="100.0" y="674.9" font-size="14" fill="#334155" text-anchor="middle">Q[i]</text>
<text x="100" y="701" font-size="9" fill="#334155" text-anchor="middle">BM × D</text>
<line x1="144" y1="667" x2="197" y2="667" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="247" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 0 轮</text>
<rect x="205" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="247.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[5]</text>
<rect x="205" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="247.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[5]</text>
<rect x="200" y="617" width="94" height="96" rx="5" fill="none" stroke="#ea580c"/>
<text x="359" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 1 轮</text>
<rect x="317" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="359.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[4]</text>
<rect x="317" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="359.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[4]</text>
<text x="471" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 2 轮</text>
<rect x="429" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="471.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[3]</text>
<rect x="429" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="471.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[3]</text>
<text x="583" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 3 轮</text>
<rect x="541" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="583.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[2]</text>
<rect x="541" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="583.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[2]</text>
<text x="695" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 4 轮</text>
<rect x="653" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="695.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[1]</text>
<rect x="653" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="695.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[1]</text>
<text x="807" y="613" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">第 5 轮</text>
<rect x="765" y="622" width="84" height="39" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="807.0" y="645.7" font-size="12" fill="#334155" text-anchor="middle">K[0]</text>
<rect x="765" y="669" width="84" height="39" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="807.0" y="692.7" font-size="12" fill="#334155" text-anchor="middle">V[0]</text>
<text x="530" y="735" font-size="10" fill="#334155" text-anchor="middle">K / V tile：64 × D</text>
<text x="60" y="782" font-size="13" fill="#0f172a" font-weight="700">②b 一轮内的 Q / S 行分片</text>
<text x="125.0" y="808" font-size="14" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i]</text>
<rect x="60" y="820" width="130" height="99" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<rect x="60" y="820.0" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="125.0" y="828.9875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="60" y="832.375" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="125.0" y="841.3625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="60" y="844.75" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="125.0" y="853.7375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="60" y="857.125" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="125.0" y="866.1125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<rect x="60" y="869.5" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="125.0" y="878.4875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="60" y="881.875" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="125.0" y="890.8625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="60" y="894.25" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="125.0" y="903.2375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="60" y="906.625" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="125.0" y="915.6125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<text x="125.0" y="942" font-size="11" fill="#334155" text-anchor="middle">128 × D</text>
<text x="296.0" y="808" font-size="14" fill="#16a34a" font-weight="700" text-anchor="middle">K[j]ᵀ</text>
<rect x="218" y="820" width="156" height="99" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="296.0" y="869" font-size="11" fill="#334155" text-anchor="middle">完整 tile</text>
<text x="296.0" y="942" font-size="11" fill="#334155" text-anchor="middle">D × 64</text>
<text x="475.0" y="808" font-size="14" fill="#dc2626" font-weight="700" text-anchor="middle">S/P</text>
<rect x="410" y="820" width="130" height="99" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<rect x="410" y="820.0" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="475.0" y="828.9875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="410" y="832.375" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="475.0" y="841.3625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="410" y="844.75" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="475.0" y="853.7375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="410" y="857.125" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="475.0" y="866.1125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<rect x="410" y="869.5" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="475.0" y="878.4875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="410" y="881.875" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="475.0" y="890.8625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="410" y="894.25" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="475.0" y="903.2375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="410" y="906.625" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="475.0" y="915.6125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<text x="475.0" y="942" font-size="11" fill="#334155" text-anchor="middle">128 × 64</text>
<text x="641.0" y="808" font-size="14" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j]</text>
<rect x="576" y="820" width="130" height="99" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="641.0" y="869" font-size="11" fill="#334155" text-anchor="middle">完整 tile</text>
<text x="641.0" y="942" font-size="11" fill="#334155" text-anchor="middle">64 × D</text>
<text x="811.0" y="808" font-size="14" fill="#ea580c" font-weight="700" text-anchor="middle">O[i]</text>
<rect x="746" y="820" width="130" height="99" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<rect x="746" y="820.0" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="811.0" y="828.9875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="746" y="832.375" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="811.0" y="841.3625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="746" y="844.75" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="811.0" y="853.7375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="746" y="857.125" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="811.0" y="866.1125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<rect x="746" y="869.5" width="130" height="12.375" rx="5" fill="#dbeafe" stroke="#94a3b8"/>
<text x="811.0" y="878.4875" font-size="8" fill="#334155" text-anchor="middle">w0</text>
<rect x="746" y="881.875" width="130" height="12.375" rx="5" fill="#ccfbf1" stroke="#94a3b8"/>
<text x="811.0" y="890.8625" font-size="8" fill="#334155" text-anchor="middle">w1</text>
<rect x="746" y="894.25" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="811.0" y="903.2375" font-size="8" fill="#334155" text-anchor="middle">w2</text>
<rect x="746" y="906.625" width="130" height="12.375" rx="5" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="811.0" y="915.6125" font-size="8" fill="#334155" text-anchor="middle">w3</text>
<text x="811.0" y="942" font-size="11" fill="#334155" text-anchor="middle">128 × D</text>
<text x="200" y="879" font-size="17" fill="#334155" text-anchor="middle">×</text>
<text x="391" y="879" font-size="17" fill="#334155" text-anchor="middle">=</text>
<text x="558" y="879" font-size="17" fill="#334155" text-anchor="middle">×</text>
<text x="725" y="879" font-size="17" fill="#334155" text-anchor="middle">=</text>
<text x="48" y="995" font-size="11" fill="#334155">②c 执行流与缓冲复用</text>
<text x="300" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 0 轮 · KV5</text>
<text x="406" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 1 轮 · KV4</text>
<text x="512" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 2 轮 · KV3</text>
<text x="618" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 3 轮 · KV2</text>
<text x="724" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 4 轮 · KV1</text>
<text x="830" y="1017" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">第 5 轮 · KV0</text>
<text x="54" y="1072" font-size="14" fill="#2563eb" font-weight="700">warp 0</text>
<text x="54" y="1093" font-size="10" fill="#334155">各自计算 Q 行</text>
<rect x="188" y="1030" width="50" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="213.0" y="1045.65" font-size="9" fill="#334155" text-anchor="middle">读 Q</text>
<rect x="188" y="1062" width="50" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="213.0" y="1077.3" font-size="8" fill="#334155" text-anchor="middle">发 K[5]</text>
<rect x="250" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="300.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[5] → V 缓冲</text>
<rect x="250" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="300.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[5][w0]</text>
<rect x="250" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="300.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[4]</text>
<rect x="250" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="300.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="356" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="406.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[4] → V 缓冲</text>
<rect x="356" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="406.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[4][w0]</text>
<rect x="356" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="406.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[3]</text>
<rect x="356" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="406.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="462" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="512.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[3] → V 缓冲</text>
<rect x="462" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="512.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[3][w0]</text>
<rect x="462" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="512.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[2]</text>
<rect x="462" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="512.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="568" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="618.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[2] → V 缓冲</text>
<rect x="568" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="618.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[2][w0]</text>
<rect x="568" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="618.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[1]</text>
<rect x="568" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="618.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="674" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="724.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[1] → V 缓冲</text>
<rect x="674" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="724.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[1][w0]</text>
<rect x="674" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="724.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[0]</text>
<rect x="674" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="724.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="780" y="1030" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="830.0" y="1042.8" font-size="8" fill="#334155" text-anchor="middle">发 V[0] → V 缓冲</text>
<rect x="780" y="1057" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="830.0" y="1069.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[0][w0]</text>
<rect x="780" y="1081" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="830.0" y="1092.8" font-size="8" fill="#334155" text-anchor="middle">不再预取</text>
<rect x="780" y="1103" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="830.0" y="1114.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<text x="54" y="1172" font-size="14" fill="#0d9488" font-weight="700">warp 1</text>
<text x="54" y="1193" font-size="10" fill="#334155">各自计算 Q 行</text>
<rect x="188" y="1130" width="50" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="213.0" y="1145.65" font-size="9" fill="#334155" text-anchor="middle">读 Q</text>
<rect x="188" y="1162" width="50" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="213.0" y="1177.3" font-size="8" fill="#334155" text-anchor="middle">发 K[5]</text>
<rect x="250" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="300.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[5] → V 缓冲</text>
<rect x="250" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="300.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[5][w1]</text>
<rect x="250" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="300.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[4]</text>
<rect x="250" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="300.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="356" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="406.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[4] → V 缓冲</text>
<rect x="356" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="406.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[4][w1]</text>
<rect x="356" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="406.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[3]</text>
<rect x="356" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="406.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="462" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="512.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[3] → V 缓冲</text>
<rect x="462" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="512.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[3][w1]</text>
<rect x="462" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="512.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[2]</text>
<rect x="462" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="512.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="568" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="618.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[2] → V 缓冲</text>
<rect x="568" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="618.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[2][w1]</text>
<rect x="568" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="618.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[1]</text>
<rect x="568" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="618.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="674" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="724.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[1] → V 缓冲</text>
<rect x="674" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="724.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[1][w1]</text>
<rect x="674" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="724.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">K 读完→发 K[0]</text>
<rect x="674" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="724.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<rect x="780" y="1130" width="100" height="20" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="830.0" y="1142.8" font-size="8" fill="#334155" text-anchor="middle">发 V[0] → V 缓冲</text>
<rect x="780" y="1157" width="100" height="20" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="830.0" y="1169.8" font-size="8" fill="#334155" text-anchor="middle">◇ QK → S[0][w1]</text>
<rect x="780" y="1181" width="100" height="18" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="830.0" y="1192.8" font-size="8" fill="#334155" text-anchor="middle">不再预取</text>
<rect x="780" y="1203" width="100" height="18" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="830.0" y="1214.8" font-size="8" fill="#334155" text-anchor="middle">◇ softmax / 修正 / PV</text>
<text x="54" y="1285" font-size="11" fill="#0f172a" font-weight="700">K buffer</text>
<rect x="188" y="1262" width="700" height="60" rx="5" fill="#ffffff" stroke="#64748b"/>
<rect x="250" y="1267" width="100" height="20" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="300.0" y="1280.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[5]</text>
<rect x="250" y="1291" width="100" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="300.0" y="1306.15" font-size="9" fill="#334155" text-anchor="middle">读完→K[4]</text>
<rect x="356" y="1267" width="100" height="20" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="406.0" y="1280.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[4]</text>
<rect x="356" y="1291" width="100" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="406.0" y="1306.15" font-size="9" fill="#334155" text-anchor="middle">读完→K[3]</text>
<rect x="462" y="1267" width="100" height="20" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="512.0" y="1280.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[3]</text>
<rect x="462" y="1291" width="100" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="512.0" y="1306.15" font-size="9" fill="#334155" text-anchor="middle">读完→K[2]</text>
<rect x="568" y="1267" width="100" height="20" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="618.0" y="1280.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[2]</text>
<rect x="568" y="1291" width="100" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="618.0" y="1306.15" font-size="9" fill="#334155" text-anchor="middle">读完→K[1]</text>
<rect x="674" y="1267" width="100" height="20" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="724.0" y="1280.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[1]</text>
<rect x="674" y="1291" width="100" height="24" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="724.0" y="1306.15" font-size="9" fill="#334155" text-anchor="middle">读完→K[0]</text>
<rect x="780" y="1267" width="100" height="50" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="830.0" y="1295.15" font-size="9" fill="#334155" text-anchor="middle">当前 K[0]</text>
<text x="54" y="1350" font-size="11" fill="#0f172a" font-weight="700">V buffer</text>
<rect x="188" y="1327" width="700" height="60" rx="5" fill="#ffffff" stroke="#64748b"/>
<rect x="250" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="300.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[5]</text>
<rect x="356" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="406.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[4]</text>
<rect x="462" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="512.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[3]</text>
<rect x="568" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="618.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[2]</text>
<rect x="674" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="724.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[1]</text>
<rect x="780" y="1332" width="100" height="50" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="830.0" y="1360.15" font-size="9" fill="#334155" text-anchor="middle">当前 V[0]</text>
<rect x="925" y="512" width="325" height="940" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="940" y="539" font-size="14" fill="#0f172a" font-weight="700">FA2：本 CTA 如何协作</text>
<text x="940" y="594" font-size="11" fill="#0f172a" font-weight="700">① 固定 Q，逐块遍历 KV</text>
<text x="940" y="627" font-size="11" fill="#334155">CTA 的任务：固定 Q[i] = 128×D</text>
<text x="940" y="654" font-size="11" fill="#334155">沿 KV 序列分成 ⌈L / B_N⌉ 个 tile</text>
<text x="940" y="681" font-size="11" fill="#334155">每轮处理一对 K[j] / V[j]，各 B_N×D</text>
<text x="940" y="708" font-size="11" fill="#334155">本图 B_N=64，示例共6轮</text>
<text x="940" y="767" font-size="11" fill="#0f172a" font-weight="700">② 同一批4个 warp 搬运和计算</text>
<text x="940" y="800" font-size="11" fill="#334155">各 warp 计算32条不同的 Q / S / O 行</text>
<text x="940" y="827" font-size="11" fill="#334155">warp 0：Q 行0–15、64–79</text>
<text x="940" y="854" font-size="11" fill="#334155">warp 1：Q 行16–31、80–95</text>
<text x="940" y="881" font-size="11" fill="#334155">各 warp 复用同一轮完整 K/V tile</text>
<text x="940" y="908" font-size="11" fill="#334155">各自搬运分片，由 copy 布局决定</text>
<text x="940" y="935" font-size="11" fill="#334155">cp.async V[j] 与当前 QK 重叠</text>
<text x="940" y="962" font-size="11" fill="#334155">K 读完→预取 K[j−1]，与后续计算重叠</text>
<text x="940" y="989" font-size="11" fill="#334155">各 warp：QK → softmax / 修正 → PV</text>
<text x="940" y="1016" font-size="11" fill="#334155">warp 发 mma.sync，Tensor Core 执行</text>
<text x="940" y="1077" font-size="11" fill="#0f172a" font-weight="700">③ Q 常驻；K / V 各一个缓冲</text>
<text x="940" y="1110" font-size="11" fill="#334155">Q 在 SMEM 中跨 KV 轮次复用</text>
<text x="940" y="1137" font-size="11" fill="#334155">K 缓冲：当前 K 读完→装下一 K</text>
<text x="940" y="1164" font-size="11" fill="#334155">V 缓冲：当前 V 读完→装下一 V</text>
<text x="940" y="1191" font-size="11" fill="#334155">消费完才覆盖；数据就绪才计算</text>
<text x="940" y="1218" font-size="11" fill="#334155">同步保证安全复用；搬运慢仍需等待</text>
<text x="940" y="1245" font-size="11" fill="#334155">此配置 Share_Q_K_smem=false</text>
<text x="940" y="1272" font-size="11" fill="#334155">Q、K、V 使用不同 SMEM 区域</text>
<text x="940" y="1299" font-size="11" fill="#334155">配置：kernel_traits.h / 启动模板</text>
<text x="940" y="1370" font-size="11" fill="#334155">读图：列是计算轮次，不是精确周期</text>
<text x="940" y="1397" font-size="11" fill="#334155">◇ 表示对应操作的数据就绪检查</text>
<rect x="30" y="1472" width="1220" height="340" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="1499" font-size="17" fill="#0f172a" font-weight="700">③ Lane tiling：计算片段 → lane 私有元素 → 矩阵指令 / 执行硬件</text>
<text x="60" y="1522" font-size="11" fill="#334155">下面展开 warp 0；32 lane 只画0/1。</text>
<rect x="60" y="1540" width="220" height="212" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="170" y="1564" font-size="12" fill="#0f172a" font-weight="700" text-anchor="middle">warp 0 的计算片段</text>
<rect x="72" y="1578" width="96" height="38" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="120.0" y="1600.15" font-size="9" fill="#334155" text-anchor="middle">Q[i][w0]</text>
<rect x="176" y="1578" width="96" height="38" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="224.0" y="1600.15" font-size="9" fill="#334155" text-anchor="middle">K[j][w0]</text>
<rect x="72" y="1629" width="96" height="38" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="120.0" y="1651.15" font-size="9" fill="#334155" text-anchor="middle">V[j][w0]</text>
<rect x="176" y="1629" width="96" height="38" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="224.0" y="1651.15" font-size="9" fill="#334155" text-anchor="middle">S[j][w0]</text>
<rect x="72" y="1680" width="96" height="38" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="120.0" y="1702.15" font-size="9" fill="#334155" text-anchor="middle">P[j][w0]</text>
<rect x="176" y="1680" width="96" height="38" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="224.0" y="1702.15" font-size="9" fill="#334155" text-anchor="middle">O[i][w0]</text>
<line x1="286" y1="1646" x2="344" y2="1646" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="315" y="1630" font-size="10" fill="#334155" text-anchor="middle">元素分配</text>
<rect x="350" y="1540" width="440" height="212" rx="5" fill="#f8fafc" stroke="#64748b"/>
<text x="362" y="1564" font-size="12" fill="#0f172a" font-weight="700">warp：32 lane（只展开 lane 0 / lane 1）</text>
<text x="362" y="1585" font-size="11" fill="#334155">各 lane 的工作元素存在各自私有寄存器中</text>
<rect x="362" y="1598" width="416" height="60" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="374" y="1616" font-size="11" fill="#0f172a" font-weight="700">lane 0 · 私有寄存器</text>
<rect x="370.0" y="1626" width="60.0" height="22" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="400.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">Q[…][lane 0]</text>
<rect x="438.0" y="1626" width="60.0" height="22" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="468.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">K[…][lane 0]</text>
<rect x="506.0" y="1626" width="60.0" height="22" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="536.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">V[…][lane 0]</text>
<rect x="574.0" y="1626" width="60.0" height="22" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="604.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">S[…][lane 0]</text>
<rect x="642.0" y="1626" width="60.0" height="22" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="672.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">P[…][lane 0]</text>
<rect x="710.0" y="1626" width="60.0" height="22" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="740.0" y="1639.8" font-size="8" fill="#334155" text-anchor="middle">O[…][lane 0]</text>
<rect x="362" y="1670" width="416" height="60" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="374" y="1688" font-size="11" fill="#0f172a" font-weight="700">lane 1 · 私有寄存器</text>
<rect x="370.0" y="1698" width="60.0" height="22" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="400.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">Q[…][lane 1]</text>
<rect x="438.0" y="1698" width="60.0" height="22" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="468.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">K[…][lane 1]</text>
<rect x="506.0" y="1698" width="60.0" height="22" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="536.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">V[…][lane 1]</text>
<rect x="574.0" y="1698" width="60.0" height="22" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="604.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">S[…][lane 1]</text>
<rect x="642.0" y="1698" width="60.0" height="22" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="672.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">P[…][lane 1]</text>
<rect x="710.0" y="1698" width="60.0" height="22" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="740.0" y="1711.8" font-size="8" fill="#334155" text-anchor="middle">O[…][lane 1]</text>
<rect x="840" y="1540" width="380" height="212" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="1030" y="1564" font-size="13" fill="#0f172a" font-weight="700" text-anchor="middle">SM 内 Tensor Core 阵列（示意）</text>
<rect x="854" y="1590" width="144" height="121" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="926" y="1613" font-size="11" fill="#1d4ed8" font-weight="700" text-anchor="middle">warp 0</text>
<text x="926" y="1643" font-size="11" fill="#1d4ed8" font-weight="700" text-anchor="middle">mma.sync</text>
<text x="926" y="1670" font-size="11" fill="#1d4ed8" font-weight="700" text-anchor="middle">寄存器操作数</text>
<line x1="794" y1="1646" x2="850" y2="1646" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="1020" y="1585" width="188" height="47" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="1114.0" y="1612.35" font-size="11" fill="#334155" text-anchor="middle">矩阵执行单元 A</text>
<rect x="1020" y="1650" width="188" height="47" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="1114.0" y="1677.35" font-size="11" fill="#334155" text-anchor="middle">矩阵执行单元 B</text>
<line x1="1001" y1="1646" x2="1016" y2="1646" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="1030" y="1740" font-size="10" fill="#334155" text-anchor="middle">S/O 结果回各 lane 寄存器</text>
<text x="60" y="1788" font-size="10" fill="#334155">两条 lane 仅作示例；片段布局随指令确定，不把每个 lane 画成连续的一整行，也不与执行单元一一映射。</text>
<rect x="30" y="1832" width="1220" height="840" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="1859" font-size="17" fill="#0f172a" font-weight="700">④ 片上存储：block tile 存在哪里 → warp / 角色 → lane 使用哪些片段</text>
<text x="60" y="1890" font-size="14" fill="#0f172a" font-weight="700">一个 CTA 的 block tile / 共享存储</text>
<text x="660" y="1890" font-size="14" fill="#0f172a" font-weight="700">线程私有寄存器：再展开两组的 lane 0 / lane 1</text>
<rect x="60" y="1908" width="560" height="55" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="1926" font-size="11" fill="#334155" font-weight="700">HBM / 显存：输入与最终输出</text>
<rect x="72" y="1934" width="128" height="22" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="136.0" y="1948.85" font-size="11" fill="#334155" text-anchor="middle">Q</text>
<rect x="208" y="1934" width="128" height="22" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="272.0" y="1948.85" font-size="11" fill="#334155" text-anchor="middle">K</text>
<rect x="344" y="1934" width="128" height="22" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="408.0" y="1948.85" font-size="11" fill="#334155" text-anchor="middle">V</text>
<rect x="480" y="1934" width="128" height="22" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="544.0" y="1948.85" font-size="11" fill="#334155" text-anchor="middle">O</text>
<line x1="340" y1="1967" x2="340" y2="1991" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="354" y="1986" font-size="11" fill="#334155">cp.async</text>
<rect x="60" y="1996" width="560" height="184" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="2018" font-size="12" fill="#0f172a" font-weight="700">CTA 共享 SMEM：完整 block tile，按布局存储</text>
<rect x="72" y="2034" width="536" height="43" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="340.0" y="2059.7" font-size="12" fill="#334155" text-anchor="middle">Q[i] · 128×D</text>
<rect x="72" y="2088" width="260" height="52" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="202.0" y="2117.85" font-size="11" fill="#334155" text-anchor="middle">K[j] · 64×D</text>
<rect x="344" y="2088" width="264" height="52" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="476.0" y="2117.85" font-size="11" fill="#334155" text-anchor="middle">V[j] · 64×D</text>
<text x="72" y="2167" font-size="10" fill="#334155">共享存储只存一份；多个计算组可以复用，不按每个 lane 复制。</text>
<line x1="340" y1="2185" x2="340" y2="2210" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="60" y="2215" width="560" height="78" rx="5" fill="#eff6ff" stroke="#93c5fd"/>
<text x="340" y="2240" font-size="12" fill="#1d4ed8" font-weight="700" text-anchor="middle">SMEM load → 寄存器 Q/K 或 P/V → mma.sync</text>
<text x="340" y="2270" font-size="11" fill="#334155" text-anchor="middle">warp 32 lane 协同</text>
<line x1="340" y1="2298" x2="340" y2="2323" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="60" y="2328" width="560" height="180" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="2352" font-size="12" fill="#0f172a" font-weight="700">寄存器：block tile 分散在各线程，并非额外共享数组</text>
<rect x="72" y="2370" width="260" height="50" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="202.0" y="2399.2" font-size="12" fill="#334155" text-anchor="middle">S/P · 128×64</text>
<rect x="344" y="2370" width="264" height="50" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="476.0" y="2399.2" font-size="12" fill="#334155" text-anchor="middle">O · 128×D</text>
<text x="72" y="2450" font-size="11" fill="#334155">S/P、O 分散在4个 warp 的 lane 寄存器</text>
<text x="72" y="2485" font-size="11" fill="#334155">在线更新 row max / row sum；最后归一化 O。</text>
<line x1="620" y1="2440" x2="653" y2="2440" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="660" y="1996" width="560" height="270" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="672" y="2019" font-size="12" fill="#0f172a" font-weight="700">warp 0：32 lane</text>
<rect x="672" y="2034" width="264" height="211" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="804" y="2054" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 0 · 私有寄存器</text>
<rect x="680" y="2068" width="248" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="804.0" y="2084.0" font-size="10" fill="#334155" text-anchor="middle">Q[…][lane 0]</text>
<rect x="680" y="2097" width="248" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="804.0" y="2113.0" font-size="10" fill="#334155" text-anchor="middle">K[…][lane 0]</text>
<rect x="680" y="2126" width="248" height="25" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="804.0" y="2142.0" font-size="10" fill="#334155" text-anchor="middle">V[…][lane 0]</text>
<rect x="680" y="2155" width="248" height="25" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="804.0" y="2171.0" font-size="10" fill="#334155" text-anchor="middle">S[…][lane 0]</text>
<rect x="680" y="2184" width="248" height="25" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="804.0" y="2200.0" font-size="10" fill="#334155" text-anchor="middle">P[…][lane 0]</text>
<rect x="680" y="2213" width="248" height="25" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="804.0" y="2229.0" font-size="10" fill="#334155" text-anchor="middle">O[…][lane 0]</text>
<rect x="944" y="2034" width="264" height="211" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="1076" y="2054" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 1 · 私有寄存器</text>
<rect x="952" y="2068" width="248" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="1076.0" y="2084.0" font-size="10" fill="#334155" text-anchor="middle">Q[…][lane 1]</text>
<rect x="952" y="2097" width="248" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="1076.0" y="2113.0" font-size="10" fill="#334155" text-anchor="middle">K[…][lane 1]</text>
<rect x="952" y="2126" width="248" height="25" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="1076.0" y="2142.0" font-size="10" fill="#334155" text-anchor="middle">V[…][lane 1]</text>
<rect x="952" y="2155" width="248" height="25" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="1076.0" y="2171.0" font-size="10" fill="#334155" text-anchor="middle">S[…][lane 1]</text>
<rect x="952" y="2184" width="248" height="25" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="1076.0" y="2200.0" font-size="10" fill="#334155" text-anchor="middle">P[…][lane 1]</text>
<rect x="952" y="2213" width="248" height="25" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="1076.0" y="2229.0" font-size="10" fill="#334155" text-anchor="middle">O[…][lane 1]</text>
<text x="672" y="2259" font-size="9" fill="#334155">MMA 操作数与结果元素</text>
<rect x="660" y="2284" width="560" height="270" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="672" y="2307" font-size="12" fill="#0f172a" font-weight="700">warp 1：32 lane</text>
<rect x="672" y="2322" width="264" height="211" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="804" y="2342" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 0 · 私有寄存器</text>
<rect x="680" y="2356" width="248" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="804.0" y="2372.0" font-size="10" fill="#334155" text-anchor="middle">Q[…][lane 0]</text>
<rect x="680" y="2385" width="248" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="804.0" y="2401.0" font-size="10" fill="#334155" text-anchor="middle">K[…][lane 0]</text>
<rect x="680" y="2414" width="248" height="25" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="804.0" y="2430.0" font-size="10" fill="#334155" text-anchor="middle">V[…][lane 0]</text>
<rect x="680" y="2443" width="248" height="25" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="804.0" y="2459.0" font-size="10" fill="#334155" text-anchor="middle">S[…][lane 0]</text>
<rect x="680" y="2472" width="248" height="25" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="804.0" y="2488.0" font-size="10" fill="#334155" text-anchor="middle">P[…][lane 0]</text>
<rect x="680" y="2501" width="248" height="25" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="804.0" y="2517.0" font-size="10" fill="#334155" text-anchor="middle">O[…][lane 0]</text>
<rect x="944" y="2322" width="264" height="211" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<text x="1076" y="2342" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 1 · 私有寄存器</text>
<rect x="952" y="2356" width="248" height="25" rx="5" fill="#dbeafe" stroke="#2563eb"/>
<text x="1076.0" y="2372.0" font-size="10" fill="#334155" text-anchor="middle">Q[…][lane 1]</text>
<rect x="952" y="2385" width="248" height="25" rx="5" fill="#dcfce7" stroke="#16a34a"/>
<text x="1076.0" y="2401.0" font-size="10" fill="#334155" text-anchor="middle">K[…][lane 1]</text>
<rect x="952" y="2414" width="248" height="25" rx="5" fill="#ede9fe" stroke="#7c3aed"/>
<text x="1076.0" y="2430.0" font-size="10" fill="#334155" text-anchor="middle">V[…][lane 1]</text>
<rect x="952" y="2443" width="248" height="25" rx="5" fill="#fee2e2" stroke="#dc2626"/>
<text x="1076.0" y="2459.0" font-size="10" fill="#334155" text-anchor="middle">S[…][lane 1]</text>
<rect x="952" y="2472" width="248" height="25" rx="5" fill="#fef3c7" stroke="#b45309"/>
<text x="1076.0" y="2488.0" font-size="10" fill="#334155" text-anchor="middle">P[…][lane 1]</text>
<rect x="952" y="2501" width="248" height="25" rx="5" fill="#ffedd5" stroke="#ea580c"/>
<text x="1076.0" y="2517.0" font-size="10" fill="#334155" text-anchor="middle">O[…][lane 1]</text>
<text x="672" y="2547" font-size="9" fill="#334155">MMA 操作数与结果元素</text>
<text x="60" y="2560" font-size="12" fill="#ea580c" font-weight="700">O 写回：lane 寄存器归一化 → SMEM 输出 staging → HBM</text>
<path d="M60,2525 L42,2525 L42,1945 L58,1945" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="60" y="2612" font-size="11" fill="#334155">FA2 / FA3 的 S/P/O 主体在寄存器；FA4 的 S/P/O 主体在 TMEM；都只保存当前所需 tile。</text>
<text x="60" y="2645" font-size="10" fill="#334155">具体 lane 元素坐标由 CuTe / 指令布局确定；图中 […] 是工作片段索引，不是连续数组行。</text>
<rect x="30" y="2692" width="1220" height="128" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="2719" font-size="15" fill="#0f172a" font-weight="700">参数：与 AMD 图相同的 B / H / L / D 观察层次</text>
<rect x="52" y="2732" width="278" height="74" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="60" y="2740" width="32" height="20" rx="5" fill="#2563eb" stroke="#2563eb"/>
<text x="76.0" y="2754.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">B</text>
<text x="103" y="2755" font-size="12" fill="#2563eb" font-weight="700">batch</text>
<text x="60" y="2780" font-size="11" fill="#334155">独立 batch 维</text>
<text x="60" y="2798" font-size="11" fill="#334155">工作数 × B</text>
<rect x="342" y="2732" width="278" height="74" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="350" y="2740" width="32" height="20" rx="5" fill="#0d9488" stroke="#0d9488"/>
<text x="366.0" y="2754.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">H</text>
<text x="393" y="2755" font-size="12" fill="#0d9488" font-weight="700">heads</text>
<text x="350" y="2780" font-size="11" fill="#334155">独立 head 维</text>
<text x="350" y="2798" font-size="11" fill="#334155">工作数 × H</text>
<rect x="632" y="2732" width="278" height="74" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="640" y="2740" width="32" height="20" rx="5" fill="#ea580c" stroke="#ea580c"/>
<text x="656.0" y="2754.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">L</text>
<text x="683" y="2755" font-size="12" fill="#ea580c" font-weight="700">sequence</text>
<text x="640" y="2780" font-size="11" fill="#334155">Q 块数 ceil(L/128)</text>
<text x="640" y="2798" font-size="11" fill="#334155">KV 循环 ceil(L/64)</text>
<rect x="922" y="2732" width="278" height="74" rx="5" fill="#ffffff" stroke="#cbd5e1"/>
<rect x="930" y="2740" width="32" height="20" rx="5" fill="#7c3aed" stroke="#7c3aed"/>
<text x="946.0" y="2754.0" font-size="11" fill="#ffffff" font-weight="700" text-anchor="middle">D</text>
<text x="973" y="2755" font-size="12" fill="#7c3aed" font-weight="700">head dim</text>
<text x="930" y="2780" font-size="11" fill="#334155">Q/K 的归约；V/O 的列</text>
<text x="930" y="2798" font-size="11" fill="#334155">影响片段大小与存储占用</text>
<rect x="30" y="2840" width="1220" height="230" rx="5" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="2867" font-size="16" fill="#0f172a" font-weight="700">NVIDIA 术语（AMD 对应项）与源码</text>
<text x="48" y="2900" font-size="12" fill="#0f172a" font-weight="700">CTA / thread block（workgroup）</text>
<text x="48" y="2922" font-size="10" fill="#334155">协作处理一个逻辑 Q work tile</text>
<text x="660" y="2900" font-size="12" fill="#0f172a" font-weight="700">warp / 32 lane（wave / 64 lane）</text>
<text x="660" y="2922" font-size="10" fill="#334155">线程执行组；lane 对应线程</text>
<text x="48" y="2955" font-size="12" fill="#0f172a" font-weight="700">SMEM（LDS） / registers（寄存器）</text>
<text x="48" y="2977" font-size="10" fill="#334155">共享存储 / 每线程私有存储</text>
<text x="660" y="2955" font-size="12" fill="#0f172a" font-weight="700">mma.sync / wgmma / tcgen05（MFMA）</text>
<text x="660" y="2977" font-size="10" fill="#334155">矩阵指令的功能类比；参与范围和操作数路径不同</text>
<text x="48" y="3019" font-size="11" fill="#334155">warpgroup = 4 warps / 128 threads；TMA 为硬件搬运；TMEM 为 Blackwell 矩阵存储。</text>
<text x="48" y="3049" font-size="9" fill="#334155">来源：flash_attention_v2/csrc/flash_attn/src：flash_fwd_launch_template.h / kernel_traits.h / flash_fwd_kernel.h</text>
</svg>
```


## 完整独立绘图生成器

```python
from pathlib import Path
from html import escape

ROOT = Path('/tmp/fa4_learn')
COLORS = {'Q': ('#dbeafe', '#2563eb'), 'K': ('#dcfce7', '#16a34a'),
          'V': ('#ede9fe', '#7c3aed'), 'S': ('#fee2e2', '#dc2626'),
          'P': ('#fef3c7', '#b45309'), 'O': ('#ffedd5', '#ea580c')}
DIM_COLORS = {'B': '#2563eb', 'H': '#0d9488', 'L': '#ea580c', 'D': '#7c3aed'}
CONFIGS = [
    dict(key='hopper_fa2', name='Hopper FA2', bm=128, bn=64,
         threads='4 warps · 128 threads', groups=['warp 0', 'warp 1'],
         sizes=['32 个 Q 行（跨两个16行段）', '32 个 Q 行（跨两个16行段）'],
         source='flash_attention_v2/csrc/flash_attn/src：flash_fwd_launch_template.h / kernel_traits.h / flash_fwd_kernel.h'),
    dict(key='hopper_fa3', name='Hopper FA3', bm=128, bn=176,
         threads='12 warps · 384 threads（此配置活跃9 warps）', groups=['consumer WG1', 'consumer WG2'],
         sizes=['Q 前64行 · warp 4–7', 'Q 后64行 · warp 8–11'],
         source='fbcode/fa3/hopper：tile_size.h / flash_fwd_launch_template.h / flash_fwd_kernel_sm90.h / mainloop_fwd_sm90_tma_gmma_ws.hpp'),
    dict(key='blackwell_fa4', name='Blackwell FA4', bm=256, bn=128,
         threads='16 warps · 512 threads（6 类工作角色）', groups=['Q0 / softmax0', 'Q1 / softmax1'],
         sizes=['Q 前128行 · softmax warp 0–3', 'Q 后128行 · softmax warp 4–7'],
         source='ads_mkl/ops/cute_dsl/fa4/src：flash_fwd_sm100.py / interface.py')
]


def generate(cfg):
    p = []
    kind, bm, bn = cfg['key'], cfg['bm'], cfg['bn']
    fa2, fa3, fa4 = [kind.endswith(s) for s in ['fa2', 'fa3', 'fa4']]

    def text(x, y, value, size=11, color='#334155', bold=False, anchor=None):
        attrs = ' font-weight="700"' if bold else ''
        if anchor:
            attrs += f' text-anchor="{anchor}"'
        p.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}"{attrs}>{escape(str(value))}</text>')

    def box(x, y, w, h, label='', fill='#ffffff', stroke='#cbd5e1', size=11):
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}"/>')
        if label:
            text(x+w/2, y+h/2+size*.35, label, size, anchor='middle')

    def chip(x, y, w, h, label, key, size=10):
        box(x, y, w, h, label, *COLORS[key], size)

    def dim_box(x, y, key, w=30, h=18):
        color = DIM_COLORS[key]
        box(x, y, w, h, fill=color, stroke=color)
        text(x+w/2, y+h/2+4, key, 11, '#ffffff', True, 'middle')

    def arrow(x1, y1, x2, y2, color='#64748b'):
        p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#arr)"/>')

    def panel(x, y, w, h, title, size=17):
        box(x, y, w, h, fill='#f8fafc')
        text(x+18, y+27, title, size, '#0f172a', True)

    def note(y, lines):
        box(925, y, 325, 940, fill='#ffffff')
        for offset, value, bold in lines:
            text(940, y+offset, value, 14 if offset == 27 else 11,
                 '#0f172a' if bold else '#334155', bold)

    p.append('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="3100" viewBox="0 0 1280 3100" font-family="Noto Sans CJK SC,Arial,sans-serif" role="img">')
    p.append(f'<title>{cfg["name"]} tiling：与 AMD 图相同的四层展开</title><desc>Block tile，CTA 内分片与执行流，lane 元素和矩阵执行，以及完整存储层级。配置为 BF16 D128 noncausal forward。</desc>')
    p.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#64748b"/></marker></defs>')
    box(0, 0, 1280, 3100, fill='#ffffff', stroke='#ffffff')
    text(40, 40, cfg['name']+' tiling：B / L / H / D 落在 kernel 的哪一层', 25, '#0f172a', True)
    text(40, 68, f'BF16 · D=128 · noncausal forward · BM={bm} · BN={bn} · '+cfg['threads'], 12)
    text(40, 91, 'FA2：标准 CUDA、dropout=0；FA3：dense / no split / no paged KV；FA4：m_block=128、Q stage=2。', 11)
    for x,key,label in [(40,'B','batch'),(195,'H','heads'),(350,'L','sequence'),(535,'D','head dim')]:
        dim_box(x,98,key,26,15)
        text(x+34,110,label,10)

    # Same left drawing / right parameter note layout as the AMD artifact.
    panel(30, 118, 880, 390, '① Block tile：原始 Q/K/V → Q 行块固定，沿 K/V 序列切块', 16)
    text(48, 168, f'一次 CTA 工作：Q[i] {bm}×D；K[j]/V[j] {bn}×D；S/P {bm}×{bn}；O[i] {bm}×D。', 11)
    text(48, 190, f'一组 (b,h)，示例 L=6×BN={6*bn}；橙框固定 Q 的第 i 块，橙色箭头表示 KV 循环 j。', 10)
    for x, w, h, key, label, shape in [(70,52,158,'Q','Q','L × D'), (185,168,55,'K','Kᵀ','D × L'),
                                       (425,180,158,'S','S / P','L × L（不整块存储）'),
                                       (700,52,158,'V','V','L × D'), (820,52,158,'O','O','L × D')]:
        yy = 302 if key == 'K' else 237
        text(x+w/2, 218, label, 20, COLORS[key][1], True, 'middle')
        for layer in [2,1]:
            box(x+layer*3, yy-layer*3, w, h, fill='#ffffff', stroke='#cbd5e1')
        box(x, yy, w, h, fill=COLORS[key][0], stroke=COLORS[key][1])
        if key == 'K':
            for col in range(6):
                box(x+col*w/6, yy, w/6, h, fill=COLORS[key][0], stroke=COLORS[key][1])
            box(x+w/6, yy, w/6, h, 'j', '#bbf7d0', '#ea580c', 11)
        elif key == 'V':
            for row in range(6):
                box(x, yy+row*h/6, w, h/6, fill=COLORS[key][0], stroke=COLORS[key][1])
            box(x, yy+h/6, w, h/6, 'j', '#ddd6fe', '#ea580c', 11)
        else:
            # Q and KV matrices represent the same sequence length.
            seq_len = 6*bn
            tile_h = h*bm/seq_len
            row_count = (seq_len+bm-1)//bm
            for row in range(row_count):
                yrow = yy+row*tile_h
                rh = min(tile_h, yy+h-yrow)
                box(x, yrow, w, rh, fill=COLORS[key][0], stroke=COLORS[key][1])
            if key == 'S':
                for col in range(1,6):
                    p.append(f'<line x1="{x+col*w/6}" y1="{yy}" x2="{x+col*w/6}" y2="{yy+h}" stroke="#dc2626" stroke-dasharray="3,3"/>')
                box(x, yy+tile_h, w, tile_h, fill='none', stroke='#ea580c')
                box(x+w/6, yy+tile_h, w/6, tile_h, '', '#fca5a5', '#ea580c', 8)
                text(x+w/4, yy+tile_h*1.35, 'j', 8, anchor='middle')
                for row in range(min(3, row_count)):
                    top = yy+row*tile_h
                    color = '#ea580c' if row == 1 else '#94a3b8'
                    arrow(x+w-8, top+tile_h*.72, x+8, top+tile_h*.72, color)
                    text(x+w-10, top+tile_h*.35,
                         'i=1 · 循环 j' if row == 1 else f'i={row} · CTA work',
                         8, color, row == 1, 'end')
            else:
                box(x, yy+tile_h, w, tile_h, 'i', *COLORS[key], 11)
                box(x, yy+tile_h, w, tile_h, fill='none', stroke='#ea580c')
        text(x+w/2, 417, shape, 11, anchor='middle')
    text(150, 337, '×', 22, anchor='middle')
    text(385, 337, '=', 22, anchor='middle')
    text(644, 274, 'S → P', 12, '#b45309', True, 'middle')
    text(644, 305, 'softmax', 10, anchor='middle')
    text(661, 337, '×', 22, anchor='middle')
    text(802, 337, '=', 22, anchor='middle')
    arrow(353, 280, 185, 280, '#ea580c')
    text(269, 269, '循环 j：5 → 0（沿 L）', 11, '#ea580c', True, 'middle')
    text(269, 384, 'K[j]ᵀ：D × BN', 11, '#16a34a', True, 'middle')
    text(515, 443, '橙行：固定 Q[i]，行内循环 j', 10, '#ea580c', True, 'middle')
    arrow(770, 394, 770, 237, '#ea580c')
    p.append('<text x="783" y="303" font-size="10" fill="#ea580c" font-weight="700" text-anchor="middle" transform="rotate(-90,783,303)">循环 j：5 → 0</text>')
    text(96, 443, '固定 Q[i]', 10, '#2563eb', True, 'middle')
    text(846, 443, '累积 O[i]', 10, '#ea580c', True, 'middle')
    oy = 237+158*bm/(6*bn)
    oh = 158*bm/(6*bn)
    p.append(f'<path d="M875,{oy+3} L896,{oy+3} L896,{oy+oh-3} L876,{oy+oh-3}" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>')
    text(48, 467, '第 j 轮：S=Q[i]K[j]ᵀ×scale → 指数权重 P → 修正已有 O 并累加 PV[j]。', 11)
    text(48, 492, '一个 CTA / Q tile' if fa2 else '一个逻辑 Q work tile / 次；persistent CTA 可连续处理多个 work tile。', 11)
    panel(925, 118, 325, 390, 'B / H / L / D 决定什么', 16)
    for y, key, value in [(178,'B','独立 batch：工作数 × B'), (219,'H','独立 head：工作数 × H'),
                           (260,'L',f'Q 行块数 ceil(L/{bm})'), (286,'',f'KV 循环数 ceil(L/{bn})'),
                           (330,'D','Q/K 的归约维；V/O 的列维')]:
        if key:
            dim_box(940, y-14, key, 26,18)
        text(970, y, value, 11)
    text(940, 370, '物理 CTA 数 ≠ 逻辑 work tile 数' if not fa2 else 'grid：B × H × ceil(L / 128)', 11, bold=True)
    text(940, 406, 'SM：调度 CTA 的计算资源', 11)
    text(940, 438, 'warp：32 lane；lane 是线程', 11)
    text(940, 470, 'Tensor Core：矩阵执行硬件', 11)

    panel(30, 512, 880, 940, '② CTA 内：先画分片，再画执行流与共享缓冲', 17)
    text(48, 560, '示例：6 个 KV block，KV5 → KV0', 11)
    text(60, 586, '②a 固定 Q，遍历 KV', 13, '#0f172a', True)
    chip(60, 622, 80, 96, 'Q[i]', 'Q', 14)
    text(100, 701, 'BM × D', 9, anchor='middle')
    arrow(144, 667, 197, 667)
    for r, j in enumerate(range(5,-1,-1)):
        x=205+112*r
        text(x+42, 613, f'第 {r} 轮', 11, bold=True, anchor='middle')
        chip(x, 622, 84, 39, f'K[{j}]', 'K', 12)
        chip(x, 669, 84, 39, f'V[{j}]', 'V', 12)
        if r == 0:
            box(x-5, 617, 94, 96, fill='none', stroke='#ea580c')
    text(530, 735, f'K / V tile：{bn} × D', 10, anchor='middle')
    text(60, 782, '②b 一轮内的 Q / S 行分片', 13, '#0f172a', True)
    tiles_2b = [(165,95,'Q','Q[i]',f'{bm} × D'), (295,135,'K','K[j]ᵀ',f'D × {bn}'),
                (455,145,'S','S/P',f'{bm} × {bn}'), (635,95,'V','V[j]',f'{bn} × D'),
                (770,106,'O','O[i]',f'{bm} × D')] if fa4 else [
                (60,130,'Q','Q[i]',f'{bm} × D'), (218,156,'K','K[j]ᵀ',f'D × {bn}'),
                (410,130,'S','S/P',f'{bm} × {bn}'), (576,130,'V','V[j]',f'{bn} × D'),
                (746,130,'O','O[i]',f'{bm} × D')]
    for x,w,key,label,shape in tiles_2b:
        text(x+w/2, 808, label, 14, COLORS[key][1], True, 'middle')
        box(x, 820, w, 99, fill=COLORS[key][0], stroke=COLORS[key][1])
        if key in ('Q','S','O'):
            if fa2:
                for band in range(8):
                    ww = band%4
                    fill=['#dbeafe','#ccfbf1','#f1f5f9','#f1f5f9'][ww]
                    box(x,820+band*99/8,w,99/8,f'w{ww}',fill,'#94a3b8',8)
            else:
                for part in range(2):
                    label2 = ('WG'+str(part+1)) if fa3 else ('Q'+str(part))
                    if fa4 and key in ('Q','S'):
                        by=820+part*49.5
                        fills = ['#dbeafe','#bfdbfe','#93c5fd','#60a5fa'] if part==0 else ['#ccfbf1','#99f6e4','#5eead4','#2dd4bf']
                        stroke = '#2563eb' if part==0 else '#0d9488'
                        if key=='Q':
                            text(x-22,by+20,f'Q{part}',12,stroke,True,'end')
                            text(x-22,by+37,'128 条 query',10,stroke,True,'end')
                            p.append(f'<path d="M{x-5},{by+2} L{x-13},{by+2} L{x-13},{by+47.5} L{x-5},{by+47.5}" fill="none" stroke="{stroke}" stroke-width="1.4"/>')
                        for warp in range(4):
                            label3 = f'query {warp*32}–{warp*32+31}' if key=='Q' else f'w{part*4+warp} · Q{part} 行{warp*32}–{warp*32+31}'
                            box(x,by+warp*49.5/4,w,49.5/4,label3,fills[warp],stroke,8)
                        box(x,by,w,49.5,fill='none',stroke=stroke)
                    else:
                        box(x, 820+part*49.5, w, 49.5, label2,
                            '#dbeafe' if part==0 else '#ccfbf1', '#94a3b8', 12)
        else:
            text(x+w/2, 869, '完整 tile', 11, anchor='middle')
        text(x+w/2, 942, shape, 11, anchor='middle')
    for index,x in enumerate([277,442,617,749] if fa4 else [200,391,558,725]):
        text(x, 879, '×' if index in [0,2] else '=', 17, anchor='middle')
    text(48, 995, '②c 执行流与缓冲复用', 11)
    for r,j in enumerate(range(5,-1,-1)):
        text(300+106*r, 1017, f'第 {r} 轮 · KV{j}', 9, bold=True, anchor='middle')
    if fa2:
        for warp,y in [(0,1030),(1,1130)]:
            text(54,y+42,f'warp {warp}',14,'#2563eb' if warp==0 else '#0d9488',True)
            text(54,y+63,'各自计算 Q 行',10)
            chip(188,y,50,25,'读 Q','Q',9)
            chip(188,y+32,50,25,'发 K[5]','K',8)
            for r,j in enumerate(range(5,-1,-1)):
                x=250+106*r
                chip(x,y,100,20,f'发 V[{j}] → V 缓冲','V',8)
                chip(x,y+27,100,20,f'◇ QK → S[{j}][w{warp}]','S',8)
                chip(x,y+51,100,18,f'K 读完→发 K[{j-1}]' if j else '不再预取','K',8)
                chip(x,y+73,100,18,'◇ softmax / 修正 / PV','P',8)
    elif fa3:
        text(54,1047,'w0 · 预取 K',11,'#0f172a',True)
        text(54,1081,'w0 · 预取 V',11,'#0f172a',True)
        text(213,1017,'启动加载',9,bold=True,anchor='middle')
        for order,(name,key) in enumerate([('Q[i]','Q'),('K[5]','K'),('V[5]','V')]):
            chip(188,1030+19*order,50,16,name,key,8)
        for r,j in enumerate(range(5,-1,-1)):
            x=250+106*r
            stage=(r+1)%2
            for yy,key in [(1030,'K'),(1064,'V')]:
                if j:
                    chip(x,yy,100,26,f'发 {key}[{j-1}] → {key} stage {stage}',key,8)
                else:
                    box(x,yy,100,26,'不再预取 '+key,size=9)
        for group,y in [(1,1106),(2,1177)]:
            text(54,y+20,f'WG{group} · w{group*4}–{group*4+3}',11,'#0f172a',True)
            text(54,y+42,'前64个 Q 行' if group==1 else '后64个 Q 行',10)
            for r,j in enumerate(range(5,-1,-1)):
                x=250+106*r
                chip(x,y,100,23,f'◇ QK → S[{j}][WG{group}]','S',8)
                chip(x,y+29,100,23,'◇ softmax / 修正 → PV','P',8)
    else:
        text(54,1047,'w14 · 预取 K',11,'#0f172a',True)
        text(54,1079,'w14 · 预取 V',11,'#0f172a',True)
        text(213,1017,'启动',9,bold=True,anchor='middle')
        for order,(name,key) in enumerate([('Q0','Q'),('K[5]','K'),('Q1','Q'),('V[5]','V')]):
            chip(188,1029+15*order,50,13,name,key,8)
        for r,j in enumerate(range(5,-1,-1)):
            x=250+106*r
            if j:
                ks,vs=(2*r+2)%3,(2*r)%3
                chip(x,1030,100,25,f'发 K[{j-1}] → slot {ks}','K',8)
                box(x,1060,100,31,fill=COLORS['V'][0],stroke=COLORS['V'][1])
                text(x+50,1072,f'等 K[{j}] 读完 / 释放',8,anchor='middle')
                text(x+50,1085,f'发 V[{j-1}] → slot {vs}',8,anchor='middle')
            else:
                box(x,1030,100,25,'不再预取 K',size=9)
                box(x,1060,100,31,'不再预取 V',size=9)
        rows=[('MMA · w12','◇ QK0 / QK1','S'),
              ('softmax0 · w0–3','◇ S(Q0) → P(Q0)','P'),('softmax1 · w4–7','◇ S(Q1) → P(Q1)','P'),
              ('correction · w8–11','◇ 修正 O','O'),('epilogue · w13','等待最终 O','O')]
        for row,(name,value,key) in enumerate(rows):
            y=1098+32*row
            text(54,y+20,name,11,'#0f172a',True)
            for r,j in enumerate(range(5,-1,-1)):
                label='归一化 O' if row==3 and r==5 else 'SMEM → HBM' if row==4 and r==5 else value
                if row==0:
                    box(250+106*r,y,100,29,fill=COLORS['S'][0],stroke=COLORS['S'][1])
                    text(300+106*r,y+11,f'QK(Q0,K{j}) / QK(Q1,K{j})',7,anchor='middle')
                    text(300+106*r,y+23,'PV(Q0) / PV(Q1)',8,anchor='middle')
                else:
                    chip(250+106*r,y,100,26,label,key,8)

    # Physical stages kept as rows, like the AMD slot drawing.
    sy=1262
    if fa2:
        rows=[('K buffer','K'),('V buffer','V')]
    elif fa3:
        rows=[('K stage 0','K'),('K stage 1','K'),('V stage 0','V'),('V stage 1','V')]
    else:
        rows=[('SMEM stage 0',None),('SMEM stage 1',None),('SMEM stage 2',None)]
    rh=34 if fa3 else 45 if fa4 else 65
    for row,(label,key) in enumerate(rows):
        y=sy+row*rh
        text(54,y+23,label,11,'#0f172a',True)
        box(188,y,700,rh-5,fill='#ffffff',stroke='#64748b')
        for r,j in enumerate(range(5,-1,-1)):
            x=250+106*r
            if fa2:
                if key=='K' and j:
                    chip(x,y+5,100,20,f'当前 K[{j}]','K',9)
                    chip(x,y+29,100,24,f'读完→K[{j-1}]','K',9)
                else:
                    chip(x,y+5,100,rh-15,f'当前 {key}[{j}]',key,9)
            elif fa3:
                st=row%2
                current=st==r%2
                jj=j if current else j-1
                if jj>=0:
                    chip(x,y+4,100,rh-13,f'{key}[{jj}]'+(' 当前' if current else ' 下一块'),key,8)
                else:
                    box(x,y+4,100,rh-13,'空闲',size=8)
            else:
                ks,vs,nks=(2*r)%3,(2*r+1)%3,(2*r+2)%3
                if row==ks:
                    if j:
                        chip(x,y+4,100,14,f'当前 K[{j}]','K',8)
                        chip(x,y+21,100,15,f'读完→预取 V[{j-1}]','V',7)
                    else:
                        chip(x,y+4,100,rh-13,f'当前 K[{j}]','K',9)
                elif row==vs: chip(x,y+4,100,rh-13,f'当前 V[{j}]','V',9)
                elif j: chip(x,y+4,100,rh-13,f'下一 K[{j-1}]','K',9)
                else: box(x,y+4,100,rh-13,'空闲',size=9)
    if fa2:
        summary=[(27,'FA2：本 CTA 如何协作',True),
                 (82,'① 固定 Q，逐块遍历 KV',True),
                 (115,'CTA 的任务：固定 Q[i] = 128×D',False),
                 (142,'沿 KV 序列分成 ⌈L / B_N⌉ 个 tile',False),
                 (169,'每轮处理一对 K[j] / V[j]，各 B_N×D',False),
                 (196,'本图 B_N=64，示例共6轮',False),
                 (255,'② 同一批4个 warp 搬运和计算',True),
                 (288,'各 warp 计算32条不同的 Q / S / O 行',False),
                 (315,'warp 0：Q 行0–15、64–79',False),
                 (342,'warp 1：Q 行16–31、80–95',False),
                 (369,'各 warp 复用同一轮完整 K/V tile',False),
                 (396,'各自搬运分片，由 copy 布局决定',False),
                 (423,'cp.async V[j] 与当前 QK 重叠',False),
                 (450,'K 读完→预取 K[j−1]，与后续计算重叠',False),
                 (477,'各 warp：QK → softmax / 修正 → PV',False),
                 (504,'warp 发 mma.sync，Tensor Core 执行',False),
                 (565,'③ Q 常驻；K / V 各一个缓冲',True),
                 (598,'Q 在 SMEM 中跨 KV 轮次复用',False),
                 (625,'K 缓冲：当前 K 读完→装下一 K',False),
                 (652,'V 缓冲：当前 V 读完→装下一 V',False),
                 (679,'消费完才覆盖；数据就绪才计算',False),
                 (706,'同步保证安全复用；搬运慢仍需等待',False),
                 (733,'此配置 Share_Q_K_smem=false',False),
                 (760,'Q、K、V 使用不同 SMEM 区域',False),
                 (787,'配置：kernel_traits.h / 启动模板',False),
                 (858,'读图：列是计算轮次，不是精确周期',False),
                 (885,'◇ 表示对应操作的数据就绪检查',False)]
    elif fa3:
        summary=[(27,'FA3：本 CTA 如何协作',True),
                 (82,'① 固定 Q，逐块遍历 KV',True),
                 (115,'CTA 的当前任务：固定 Q[i] = 128×D',False),
                 (142,'沿 KV 序列分成 ⌈L / B_N⌉ 个 tile',False),
                 (169,'每轮处理一对 K[j] / V[j]，各 B_N×D',False),
                 (196,'本图 B_N=176，示例共6轮',False),
                 (255,'② producer / consumer 分工',True),
                 (288,'CTA 共12个 warp；此配置9个工作',False),
                 (315,'w0：发 TMA；w1–3 退出',False),
                 (342,'WG1（w4–7）：前64条 Q 行',False),
                 (369,'WG2（w8–11）：后64条 Q 行',False),
                 (396,'两组复用同一轮完整 K/V tile',False),
                 (423,'每组执行 QK → softmax / 修正 → PV',False),
                 (450,'WGMMA：由4个 warp 共同参与',False),
                 (477,'w0 在目标 stage 空闲时预取下一块',False),
                 (565,'③ K / V 各两个独立 stage',True),
                 (598,'各有当前块、下一块的两个缓冲槽',False),
                 (625,'消费完才覆盖；数据就绪才计算',False),
                 (652,'K / V pipeline 独立；V 可滞后 K',False),
                 (679,'下方示意 K/V 均到齐的缓冲快照',False),
                 (706,'同步保证安全复用；搬运慢仍需等待',False),
                 (733,'Q 另存 SMEM；配置 kStages=2',False),
                 (760,'配置：flash_fwd_launch_template.h',False),
                 (787,'K/V pipeline：mainloop_fwd_sm90*',False),
                 (858,'读图：列是工作对应；预取可领先',False),
                 (885,'◇ 表示对应操作的数据就绪检查',False)]
    else:
        summary=[(27,'FA4：本 CTA 如何协作',True),
                 (82,'① 固定 Q，逐块遍历 KV',True),
                 (115,'CTA 的当前任务：固定 Q[i] = 256×D',False),
                 (142,'沿 KV 序列分成 ⌈L / B_N⌉ 个 tile',False),
                 (169,'每轮处理一对 K[j] / V[j]，各 B_N×D',False),
                 (196,'本图 B_N=128，示例共6轮',False),
                 (255,'② 15 个工作 warp 按角色协作',True),
                 (288,'CTA 共16个 warp；w15 空闲',False),
                 (315,'8个 softmax warp：每个负责32条 Q 行',False),
                 (342,'w0–3 → Q0 的128行对应的 S/P',False),
                 (369,'w4–7 → Q1 的128行对应的 S/P',False),
                 (396,'分的是 query 行，不是 head dim D',False),
                 (423,'w12：发 QK / PV 矩阵指令',False),
                 (450,'w8–11：修正 O / 最终归一化',False),
                 (477,'w13：写回 O；w14：发 TMA 搬运',False),
                 (565,'③ 三个 KV slot：重叠搬运与计算',True),
                 (598,'每个 slot 存一个 K 或 V tile；Q 另存',False),
                 (625,'可容纳：当前 K + 当前 V + 下一 K',False),
                 (652,'当前 K 读完 → 原 slot 装下一 V',False),
                 (679,'消费完才覆盖；数据就绪才计算',False),
                 (706,'同步保证安全复用；搬运慢仍需等待',False),
                 (733,'BF16：self.kv_stage = 3',False),
                 (760,'配置：flash_fwd_sm100.py',False),
                 (787,'           _setup_attributes()',False),
                 (858,'读图：列是计算轮次；预取可领先',False),
                 (885,'◇ 表示对应操作的数据就绪检查',False)]
    note(512,summary)

    # Same three-column tile → two lanes → instruction / unit array layout.
    panel(30,1472,1220,340,'③ Lane tiling：计算片段 → lane 私有元素 → 矩阵指令 / 执行硬件',17)
    text(60,1522,'下面展开 warp 0；32 lane 只画0/1。' if fa2 else '下面展开 consumer WG1 的 warp 4；WGMMA 的参与范围是整个4-warp WG。' if fa3 else
         '下面展开 softmax0 的 warp 0；它处理 S/P，矩阵指令由另一个 MMA warp 发出。',11)
    box(60,1540,220,212)
    text(170,1564,'warp 0 的计算片段' if fa2 else 'WG1 的矩阵片段' if fa3 else 'Q0 对应的 S/P 片段',12,'#0f172a',True,'middle')
    labels=['Q','K','V','S','P','O'] if fa2 else ['S','P','O'] if fa3 else ['S','P']
    for c,key in enumerate(labels):
        idx='i' if key in ('Q','O') else 'j'
        chip(72+(c%2)*104,1578+(c//2)*51,96,38,f'{key}[{idx}][w0]' if fa2 else f'{key}[{idx}][WG1]' if fa3 else f'{key}[{idx}](Q0)',key,9)
    arrow(286,1646,344,1646)
    text(315,1630,'元素分配',10,anchor='middle')
    box(350,1540,440,212,fill='#f8fafc',stroke='#64748b')
    text(362,1564,'warp：32 lane（只展开 lane 0 / lane 1）',12,'#0f172a',True)
    text(362,1585,'各 lane 的工作元素存在各自私有寄存器中',11)
    for lane,y in [(0,1598),(1,1670)]:
        box(362,y,416,60)
        text(374,y+18,f'lane {lane} · 私有寄存器',11,'#0f172a',True)
        ls=labels+([] if fa2 else ['m/l'])
        cw=(400-8*(len(ls)-1))/len(ls)
        for c,key in enumerate(ls):
            chip(370+c*(cw+8),y+28,cw,22,f'{key}[…][lane {lane}]',key if key in COLORS else 'P',8)
    box(840,1540,380,212)
    text(1030,1564,'SM 内 Tensor Core 阵列（示意）',13,'#0f172a',True,'middle')
    box(854,1590,144,121,fill='#eff6ff',stroke='#93c5fd')
    for y,label in [(1613,'warp 0' if fa2 else 'consumer WG1' if fa3 else 'MMA warp 12'),
                    (1643,'mma.sync' if fa2 else 'wgmma' if fa3 else 'tcgen05'),
                    (1670,'寄存器操作数' if fa2 else 'SMEM / 寄存器' if fa3 else 'SMEM / TMEM')]:
        text(926,y,label,11,'#1d4ed8',True,'middle')
    if fa2:
        arrow(794,1646,850,1646)
    else:
        text(804,1627,'累加器' if fa3 else 'P→TMEM',8,anchor='middle')
        arrow(850,1646,794,1646) if fa3 else arrow(794,1646,850,1646)
    for unit,y in [('A',1585),('B',1650)]:
        box(1020,y,188,47,'矩阵执行单元 '+unit,'#eff6ff','#93c5fd',11)
    arrow(1001,1646,1016,1646)
    text(1030,1740,'S/O 结果回各 lane 寄存器' if not fa4 else 'S/O 结果写 TMEM，不回 issuer lane',10,anchor='middle')
    text(60,1788,'单元 A/B 仅表示硬件阵列，不表示一个 lane 对应一个单元；FA4 softmax warp 与 MMA warp 是不同角色。' if fa4 else
         '两条 lane 仅作示例；片段布局随指令确定，不把每个 lane 画成连续的一整行，也不与执行单元一一映射。',10)

    # Same nested block → warp / role → lane memory hierarchy.
    panel(30,1832,1220,840,'④ 片上存储：block tile 存在哪里 → warp / 角色 → lane 使用哪些片段',17)
    text(60,1890,'一个 CTA 的 block tile / 共享存储',14,'#0f172a',True)
    text(660,1890,'线程私有寄存器：再展开两组的 lane 0 / lane 1',14,'#0f172a',True)
    box(60,1908,560,55)
    text(72,1926,'HBM / 显存：输入与最终输出',11,bold=True)
    for c,key in enumerate(['Q','K','V','O']):chip(72+136*c,1934,128,22,key,key,11)
    arrow(340,1967,340,1991)
    text(354,1986,'cp.async' if fa2 else 'TMA',11)
    box(60,1996,560,184)
    text(72,2018,'CTA 共享 SMEM：完整 block tile，按布局存储',12,'#0f172a',True)
    chip(72,2034,536,43,f'Q[i] · {bm}×D' if not fa4 else 'Q0 / Q1 · 各128×D','Q',12)
    if fa2:
        chip(72,2088,260,52,'K[j] · 64×D','K',11)
        chip(344,2088,264,52,'V[j] · 64×D','V',11)
    elif fa3:
        for st in range(2):
            chip(72+272*st,2088,260,25,f'K stage {st} · 176×D','K',10)
            chip(72+272*st,2120,260,25,f'V stage {st} · 176×D','V',10)
    else:
        for st in range(3):
            box(72+180*st,2088,176,60,f'stage {st}：K 或 V','#f1f5f9','#64748b',11)
    text(72,2167,'共享存储只存一份；多个计算组可以复用，不按每个 lane 复制。',10)
    arrow(340,2185,340,2210)
    box(60,2215,560,78,fill='#eff6ff',stroke='#93c5fd')
    text(340,2240,'SMEM load → 寄存器 Q/K 或 P/V → mma.sync' if fa2 else
         'WGMMA QK：SMEM Q/K；PV：寄存器 P + SMEM V' if fa3 else
         'tcgen05 QK：SMEM Q/K；PV：TMEM P + SMEM V',12,'#1d4ed8',True,'middle')
    text(340,2270,'warp 32 lane 协同' if fa2 else 'warpgroup 4 warps / 128 threads 协同' if fa3 else 'MMA warp 12 发指令；Tensor Core 异步执行',11,anchor='middle')
    arrow(340,2298,340,2323)
    box(60,2328,560,180)
    text(72,2352,'TMEM：S/P 与 O 的 block tile' if fa4 else '寄存器：block tile 分散在各线程，并非额外共享数组',12,'#0f172a',True)
    chip(72,2370,260,50,f'S/P · {bm}×{bn}','S',12)
    chip(344,2370,264,50,f'O · {bm}×D','O',12)
    text(72,2450,'S/P、O 分散在4个 warp 的 lane 寄存器' if fa2 else
         '每个 consumer WG：S/P 64×176，O 64×D' if fa3 else
         'S/P 和 O 各对应 Q0 / Q1 两个区域，按角色读写',11)
    text(72,2485,'在线更新 row max / row sum；最后归一化 O。',11)
    arrow(620,2440,653,2440)
    if fa2:
        roles=[('warp 0：32 lane',['Q','K','V','S','P','O']),('warp 1：32 lane',['Q','K','V','S','P','O'])]
    elif fa3:
        roles=[('consumer WG1 / warp 4：32 lane',['S','P','O','m/l']),('consumer WG2 / warp 8：32 lane',['S','P','O','m/l'])]
    else:
        roles=[('softmax0 / warp 0：32 lane',['S','P','row max','row sum']),('correction / warp 8：32 lane',['O','修正系数','归一化系数'])]
    for ri,(label,keys) in enumerate(roles):
        y=1996+288*ri
        box(660,y,560,270,fill='#f8fafc')
        text(672,y+23,label,12,'#0f172a',True)
        for lane in range(2):
            x=672+272*lane
            box(x,y+38,264,211)
            text(x+132,y+58,f'lane {lane} · 私有寄存器',11,bold=True,anchor='middle')
            ch=25 if len(keys)==6 else 32
            for row,key in enumerate(keys):
                value=f'{key}[…][lane {lane}]' if key in COLORS else key
                chip(x+8,y+72+row*(ch+4),248,ch,value,key if key in COLORS else 'P',10)
        text(672,y+263,'MMA 操作数与结果元素' if fa2 else 'WGMMA 累加器与 P 操作数' if fa3 else '临时读写 TMEM 工作片段' if ri==0 else '读 TMEM O → 修正 / 最终归一化 → 输出',9)
    text(60,2560,'O 写回：lane 寄存器归一化 → SMEM 输出 staging → HBM' if not fa4 else
         'O 写回：TMEM → correction lane 寄存器归一化 → SMEM → epilogue → HBM',12,'#ea580c',True)
    p.append('<path d="M60,2525 L42,2525 L42,1945 L58,1945" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>')
    text(60,2612,'FA2 / FA3 的 S/P/O 主体在寄存器；FA4 的 S/P/O 主体在 TMEM；都只保存当前所需 tile。',11)
    text(60,2645,'具体 lane 元素坐标由 CuTe / 指令布局确定；图中 […] 是工作片段索引，不是连续数组行。',10)

    panel(30,2692,1220,128,'参数：与 AMD 图相同的 B / H / L / D 观察层次',15)
    for x,key,name,line1,line2 in [
        (60,'B','batch','独立 batch 维','工作数 × B'),
        (350,'H','heads','独立 head 维','工作数 × H'),
        (640,'L','sequence',f'Q 块数 ceil(L/{bm})',f'KV 循环 ceil(L/{bn})'),
        (930,'D','head dim','Q/K 的归约；V/O 的列','影响片段大小与存储占用')]:
        box(x-8,2732,278,74,fill='#ffffff')
        dim_box(x,2740,key,32,20)
        text(x+43,2755,name,12,DIM_COLORS[key],True)
        text(x,2780,line1,11)
        text(x,2798,line2,11)
    panel(30,2840,1220,230,'NVIDIA 术语（AMD 对应项）与源码',16)
    for x,y,label,desc in [(48,2900,'CTA / thread block（workgroup）','协作处理一个逻辑 Q work tile'),
                           (660,2900,'warp / 32 lane（wave / 64 lane）','线程执行组；lane 对应线程'),
                           (48,2955,'SMEM（LDS） / registers（寄存器）','共享存储 / 每线程私有存储'),
                           (660,2955,'mma.sync / wgmma / tcgen05（MFMA）','矩阵指令的功能类比；参与范围和操作数路径不同')]:
        text(x,y,label,12,'#0f172a',True)
        text(x,y+22,desc,10)
    text(48,3019,'warpgroup = 4 warps / 128 threads；TMA 为硬件搬运；TMEM 为 Blackwell 矩阵存储。',11)
    text(48,3049,'来源：'+cfg['source'],9)
    p.append('</svg>')
    (ROOT/(kind+'_tiling.svg')).write_text('\n'.join(p))


for config in CONFIGS:
    if config['key'] == 'hopper_fa2':
        generate(config)
print('Generated AMD-layout NVIDIA diagrams')
```


## 完整PNG renderer

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
