title: AMD TLX FlashAttention tiling：最终知识点、完整 SVG 与 PNG renderer
doc_type: paste
content: title: AMD TLX FlashAttention tiling：最终知识点、完整 SVG 与 PNG renderer

content: # AMD TLX FlashAttention tiling：最终学习上下文

## 1. 范围与参考

这份上下文整理了当前会话确认的知识点，以及最终 amd_tiling 图的完整 SVG 源码。

- AMD：MI350X / gfx950，当前 TLX FlashAttention forward kernel。
- NVIDIA：当前 Blackwell FA4 kernel 的对照示例；不是对全部 NVIDIA / AMD kernel 的通用结论。
- 画法参考：https://www.internalfb.com/intern/paste/P2547752777/
- AMD 源码：fbcode/ads_mkl/ops/tlx/tlx_flash_attention_gfx950.py
- Blackwell 源码：fbcode/ads_mkl/ops/cute_dsl/fa4/src/flash_fwd_sm100.py
- 源码读取与图更新日期：2026-10-15。kernel 参数和角色分工会随实现版本改变。

## 2. 四层关系与术语

workgroup / CTA 的 block tile → wave / warp 使用的片段 → lane 私有寄存器中的元素。

- workgroup（NVIDIA CTA / thread block）：协作线程块。
- wave / wavefront（NVIDIA warp）：线程执行组；这里 AMD 1 wave = 64 个线程，NVIDIA 1 warp = 32 个线程。
- lane（NVIDIA lane / thread）：执行组中的一个线程。
- CU（NVIDIA SM，功能层级对应）：容纳和执行线程块的计算单元。
- XCD（无直接对应的 NVIDIA 编程层级）：AMD 计算 chiplet，内含多个 CU。
- LDS（NVIDIA SMEM / shared memory）：workgroup 的共享片上存储。
- 寄存器（NVIDIA registers）：线程 / lane 私有；wave 的寄存器框只是对各线程的汇总。
- MFMA = Matrix Fused Multiply-Add：AMD 矩阵乘加指令，功能上对应 NVIDIA MMA；Blackwell 此实现使用 tcgen05。MFMA 不是 wave、warp 或线程组。
- HBM：显存，global memory 的物理后端。

当前 AMD 配置是 4 或 8 个 wave，即 256 或 512 个线程。“64”是一个 wave 的线程数，不是有 64 个 wave。

AMD 可以做 wave 角色分工；这份 kernel 让同一批 wave 参与搬运、矩阵乘和 softmax / 修正。不能从是否有 NVIDIA 风格的 warpgroup，直接推出是否能分工，也不能从组数判断哪种实现更优。

## 3. Block tiling：整个 workgroup 负责什么

固定 (batch, head) 后，Q / K / V 是二维矩阵。

- 一个 workgroup 固定一个 Q 行块 Q[i]，尺寸 BM × D。
- 它循环处理全部 K/V block：K[0]/V[0]、K[1]/V[1]、……。
- 当前 AMD BLOCK_N = 64，所以每个 K[j]、V[j] 都是 64 × D。
- 第 j 轮：Q[i] × K[j]^T → S[j]，S 的逻辑 block tile 为 BM × 64。
- softmax / 在线修正产生本轮权重 P[j]，逻辑尺寸同样是 BM × 64。
- P[j] × V[j] 产生 BM × D 的贡献，累积到同一份 O[i]。
- Q[i] 在循环中复用；O 与每行的 m/l 状态持续更新。
- 不在内存中完整构造 L × L 的 S/P 矩阵。

图 section 1 的 128 × D 与后面 BM × D 是同一层 Q block tile：该示例 BM=128。但 BM 不由 D 单独决定；当前 pick_config 会按 D、batch、序列长度等选择 BM=128/256 和 num_warps=4/8。BLOCK_N=64 是此实现的配置选择，不能当成 AMD 硬件的通用 tile 限制。

原始 K[j] 形状 [N,D]（这里 N=64）。计算视图 K[j]^T 为 [D,N]，所以 N 从原始 K 的 dim0 变为 K^T 的 dim1。图 2b 用 Q 的行方向、K^T 的 N 列方向、V 的 D 列方向来示意矩阵计算分块；这不是实际编译后的 wave/lane 元素布局。

## 4. Wave / lane 分片的标记及边界

图采用显式方括号，避免 Unicode 下标字体缺字。

- K[j]：第 j 个完整 K block tile。
- K[j][w]：wave w 使用 / 搬运的逻辑部分，具体含义由所在框决定。
- K[j][w][lane]：该 wave 的该 lane 所持有的工作元素。
- Q[i][w][lane]、O[i][w][lane]：i 是 Q block 编号。
- S[j][w][lane]、P[j][w][lane]：j 是本轮 K/V block 编号；固定 Q block i 在图中省略。

这些是教学图的“元素归属”记号，不是源码真的存在的三级 / 四级数组，也不是一个标记就代表一个标量。

实际布局可能同时涉及两个维度、非连续元素与复制。K/V 的搬运分配和矩阵乘操作数读取分配不一定相同。不同 wave 可读取重叠 K/V 元素，为不同输出部分服务；因此不能把 K[j][w] 理解为 wave w 独占的一块 LDS，也不能把 Q/K/V/S 的色带解释为实际线程连续行 / 列分配。

lane tiling 图是逻辑示意，没有从编译产物验证逐元素坐标。不要把它进一步解释成精确的 MFMA fragment layout。

## 5. 64 个 lane 各自做什么

每个 lane 持有自己的寄存器元素；同一个 wave 中的 lane 通常执行同一条指令，操作各自的数据。

- 发起各自负责的显存读取。
- 从 LDS 读取计算需要的 K/V 元素到寄存器。
- 协同执行 MFMA：共同提供寄存器操作数，并分别持有结果元素。
- 用其他指令计算 max / sum / exp 等 softmax 操作；必要时交换数据、进行归约。
- 累积输出并最终写回。

只有矩阵乘加使用 MFMA，搬运与 softmax 不是都“在 MFMA 上执行”。

MFMA 是 wave 指令，CU 内的矩阵执行单元是硬件。section 3 只画两个 lane，以及两个矩阵执行单元作为阵列示意；没有确认实际单元数量或具体调度映射，不能理解成 lane 0 对应 unit A、lane 1 对应 unit B。

## 6. 存储位置与访问路径

### AMD 当前 TLX kernel

- Q：HBM → tl.load → lane 私有寄存器，循环前读取并复用；不先放 LDS。
- K/V：HBM → async_load → 共享 LDS → LDS load → lane 寄存器 → MFMA。
- K 的 local_trans / 转置视图用于后续读取与 dot；不是要求 HBM 中保存转置 K。
- S：MFMA 结果分散在 lane 寄存器中。
- P：softmax 的本轮权重分散在 lane 寄存器中，转换为矩阵乘所需类型后参与 PV。
- O / acc、m/l：分散在线程寄存器中，跨 KV 迭代更新；最后 O 除以归一化因子再写 HBM。

不同 wave 的线程可以同时访问 LDS，受共享带宽、bank conflict 和硬件队列等限制。不能直接任意读取其他线程 / wave 的寄存器；wave 内可通过特定交换 / 归约指令协作，跨 wave 通常通过共享存储等方式交换。

关键区别：“wave 能执行 LDS 读写指令”，与“MFMA 指令直接从 LDS 取操作数”是两件事。这里 MFMA 的矩阵操作数来自寄存器。

### Blackwell FA4 对照示例（BF16）

- Q/K/V：HBM → TMA → SMEM。
- 此图 CTA tile 为 256 × 128，Q 分为 Q0/Q1 两个 128 × D 子块。
- tcgen05 的 QK 操作数使用 SMEM 的 Q/K。
- S/P block tile 与 O 累积的主要存储在 TMEM；此处 S/P 两组分别对应 Q0/Q1，不是 AMD 的 KV 迭代编号。
- PV 使用 TMEM 的 P 和 SMEM 的 V；矩阵乘结果写入 TMEM。
- softmax / correction / epilogue 的 lane 在各自寄存器中处理按角色需要的 S/P/O 工作元素、m/l 与修正量；不是所有角色同时持有图中全部元素。
- 最终输出回 HBM。图的写回箭头概括结果去向，没有展开每条 epilogue 指令 / staging 步骤。

## 7. 异步搬运、两个 slot 与执行顺序

双缓冲是代码配置：NUM_BUFFERS=2，K 和 V 各有两个 LDS 缓冲；把对应 K/V 缓冲合称一个逻辑 slot。

- 当前 slot：j % 2。
- 下一 slot：(j+1) % 2。
- prologue 先发 K[0]/V[0] 到 slot 0。
- 稳态源码顺序：等待当前 KV → LDS 读取当前 KV 到寄存器 → 发下一 KV 到另一槽 → QK → softmax / 在线修正 → PV。
- 末块等待、计算，不再预取；最后写 O。

例子：

- 第 0 轮：slot 0 的 K[0]/V[0] 已就绪；往 slot 1 写 K[1]/V[1]。
- 第 1 轮：slot 1 的 K[1]/V[1] 已就绪；往已用完的 slot 0 写 K[2]/V[2]。
- 第 2 轮：slot 0 的 K[2]/V[2] 已就绪；往 slot 1 写 K[3]/V[3]。

两个 slot 是 workgroup 全部 wave 共享的，用于容纳当前和下一对完整 KV block。slot 数量不与 wave 或 lane 数量一一对应；不会为每个 wave 单独复制一对缓冲。

图中“已就绪”表示本轮所需数据已到达。“写入 → 就绪”表示下一块在这一轮期间传输，完成时刻不固定；图不是等长周期或精确的时钟 trace。当前 KV 被读进寄存器后，LDS 槽也可不再被本轮计算占用。

图中 ◇ 紧靠 S 前表示数据就绪依赖；实际源代码先等待、读当前 KV，再发下一块。不要把图标位置解释为指令严格晚于下一块发起。

wave 发请求，访存硬件执行后续传输，因此可与独立计算重叠。wave 0 的 K1[0]/V1[0]、wave 1 的 K1[1]/V1[1] 是搬运部分标记；每个可能展开成多条物理请求。每个 wave 逻辑上先发 K 再发 V，但跨 wave 请求可交错，实际完成顺序不固定。

当前实现只预取下一对 KV；这不是硬件最大 outstanding async load 数量。精确并发请求上限未验证，不能用 4/8 个 wave、2 个 slot 或两次 tile 级 async_load 来推断。吞吐还取决于带宽、延迟、队列、访问合并等。

BF16 双缓冲的 K/V LDS 总量 = 2 个槽 × 2 个矩阵 × 64 × D × 2 字节。D=128 时 64 KiB，D=256 时 128 KiB；当前资源检查使用每 CU 160 KiB LDS 上限。能否驻留多个 workgroup 还要看其他资源。

## 8. S、M/C、P、PV 的含义

- S：QK^T 的本轮注意力分数（包括 kernel 中的缩放 / mask 处理）。
- M/C：图中的阶段简称，覆盖 softmax 的 max、exp、sum 和在线状态修正，不是两个固定的源码 tensor。
- P：本轮 exp(score - running_max) 权重；在线算法此时不一定已经除以最终归一化因子。
- PV：第二次矩阵乘，得到本轮对 O 的贡献。
- O：累积结果；旧 O 会按 running max 的变化 rescale，最后归一化。

执行流最后一个计算框写 PV，不再误标成“P=PV”。同名 S0[0]/S0[1] 是第 0 轮不同 wave 的工作部分，不表示两次完整重复计算。

## 9. Blackwell 六类角色的正确理解

当前源码的 working role classes 是：

- softmax0：warp 0–3。
- softmax1：warp 4–7。
- correction：warp 8–11。
- MMA：warp 12。
- epilogue：warp 13。
- load：warp 14。
- warp 15 为 empty；CTA 共 16 个 warp。

“六组”指六类软件角色，不是六个 warp，也不是六个硬件 warpgroup。

专门的 load warp 发 TMA 请求并维护流水线协作，bulk transfer 由硬件做；专用 warp 是这份实现的组织方式，不是 TMA 硬件必须单独占一个 warp 的一般规则。

Blackwell tcgen05 异步矩阵乘与 TMEM 有利于角色分工；AMD 此实现用 MFMA、寄存器和 LDS，让同一批 wave 完成多个阶段。硬件能力与实现取舍共同决定组织方式，优劣需要同输入下测吞吐和资源占用。

## 10. 最终图的结构和用户偏好

1. section 1：整体 Q/K^T/S/P/V/O 与 workgroup block tiling；矩阵名加大加粗放框外。
2. section 2a：固定 Q 行块，循环 KV0…KV5。
3. section 2b：第 0 轮的 wave 分片逻辑示意，同时画出共享 LDS 的两个槽。
4. section 2 执行流：两条 wave 行、六轮 KV、单独 K/V 请求、就绪标记、S/M/C/PV；去掉轮次之间无必要的箭头。下方两条大框是持续复用的真实逻辑 slot，每列展示槽内容与写入 / 就绪状态。
5. section 3：只竖向展开 lane 0 / lane 1，连接同一 wave 的 MFMA 指令与 CU 内矩阵单元阵列示意。
6. section 4：存储图，AMD 与 Blackwell 对照；每个 wave / warp 只示意 lane 0 / lane 1 的私有寄存器。
7. 后面保留参数总结和单独的术语对应框。simple vs prefetch 独立 section 已删除。

偏好：

- 中文解释中，AMD 术语加 NVIDIA 对应项括号；图中统一映射放最后术语框，不在每个位置重复。
- 每 stage 内说明相关 Blackwell 区别，不额外增加结尾“差异总结”。
- 少写重复注解；用实体 box、slot 内容、箭头表达关系。K/V 分开画 box。
- 避免文字重叠、裁切和 Unicode 下标缺字；索引用明确方括号。
- PNG 为主要阅读产物，保留完整 SVG 与生成 / 渲染脚本。

## 11. 本地产物与 SVG → PNG 转换

最终 SVG：1280 × 3370；PNG：2560 × 6740，2 倍分辨率。

当前会话本地产物：

- /tmp/fa4_learn/amd_tiling.svg
- /tmp/fa4_learn/amd_tiling.png
- /tmp/fa4_learn/amd_tiling_section1.png … amd_tiling_section4.png
- /tmp/fa4_learn/gen_amd_tiling.py
- /tmp/fa4_learn/draw_storage_section.py
- /tmp/fa4_learn/draw_lane_section.py
- /tmp/fa4_learn/draw_wave_section.py
- /tmp/fa4_learn/render_png.py
- /tmp/fa4_learn/crop_sections.py

生成链：draw_storage_section.py 会调用 draw_lane_section.py；然后 draw_wave_section.py 生成总 SVG generator；执行 gen_amd_tiling.py 写 SVG；最后 render_png.py 写 PNG。

PNG 使用系统 librsvg + Cairo，通过 Python ctypes 调用。使用 /usr/bin/python3；此 devserver 默认 python3 加载系统库曾失败。需要 librsvg-2.so.2、libcairo.so.2 和 libgobject-2.0.so.0。

其他 session 复用时，可把下方完整 SVG 和 renderer 保存到自己的路径，修改 renderer 输入 / 输出路径，再执行：

```bash
/usr/bin/python3 render_png.py
```

以下附完整 renderer 和最终 SVG，SVG 不是节选。

## 12. 完整 PNG renderer

```python
import ctypes as c
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
p=Path('/tmp/fa4_learn/amd_tiling.svg');root=ET.parse(p).getroot();w=int(root.attrib['width']);h=int(root.attrib['height'])
err=c.c_void_p();handle=R.rsvg_handle_new_from_file(str(p).encode(),c.byref(err));assert handle and not err.value
surface=C.cairo_image_surface_create(0,w*2,h*2);ctx=C.cairo_create(surface);C.cairo_scale(ctx,2,2)
assert R.rsvg_handle_render_document(handle,ctx,c.byref(Rect(0,0,w,h)),c.byref(err))
assert C.cairo_surface_write_to_png(surface,b'/tmp/fa4_learn/amd_tiling.png')==0
C.cairo_destroy(ctx);C.cairo_surface_destroy(surface);G.g_object_unref(handle)
print('Rendered PNG',w*2,h*2)

```

## 13. 完整最终 SVG 源码

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="3370" viewBox="0 0 1280 3370" font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, Helvetica, Arial, sans-serif">
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 z" fill="#475569"/>
    </marker>
    <style>
      .lvl { fill:#f8fafc; stroke:#cbd5e1; stroke-width:1.5; rx:10; }
      .ttl { font-size:17px; font-weight:700; fill:#0f172a; }
      .txt { font-size:13px; fill:#334155; }
      .sm  { font-size:11px; fill:#475569; }
      .mono{ font-family: Menlo, Consolas, monospace; font-size:12px; fill:#0f172a; }
      .note{ fill:#ffffff; stroke:#e2e8f0; stroke-width:1; rx:8; }
      .chipB{ fill:#2563eb; } .chipH{ fill:#0d9488; } .chipL{ fill:#ea580c; } .chipD{ fill:#7c3aed; }
      .chipT{ font-size:12px; font-weight:700; fill:#ffffff; }
    </style>
  </defs>
  <rect x="0" y="0" width="1280" height="3370" fill="#ffffff"/>
  <text x="40" y="42" font-size="24" font-weight="700" fill="#0f172a">AMD MI350X (gfx950) TLX tiling：B / L / H / D 落在 kernel 的哪一层</text>
  <text x="40" y="68" class="txt">输入：Q [B, H, L, D]　K、V 同形（TLX 按 4D [B,H,S,D] 传参）　·　TLX: BLOCK_M=128/256，BLOCK_N=64，num_warps=4/8，NUM_BUFFERS=2　·　来源 tlx_flash_attention_gfx950.py</text>
  <!-- legend chips -->
  <g transform="translate(40,84)">
    <rect class="chipB" width="34" height="20" rx="4"/><text x="17" y="14" class="chipT" text-anchor="middle">B</text>
    <text x="42" y="14" class="sm">batch</text>
    <rect class="chipH" x="100" width="34" height="20" rx="4"/><text x="117" y="14" class="chipT" text-anchor="middle">H</text>
    <text x="142" y="14" class="sm">heads</text>
    <rect class="chipL" x="200" width="34" height="20" rx="4"/><text x="217" y="14" class="chipT" text-anchor="middle">L</text>
    <text x="242" y="14" class="sm">seq length</text>
    <rect class="chipD" x="320" width="34" height="20" rx="4"/><text x="337" y="14" class="chipT" text-anchor="middle">D</text>
    <text x="362" y="14" class="sm">head_dim</text>
  </g>
  <!-- ================= Level 1 ================= -->
  <rect class="lvl" x="30" y="118" width="880" height="386"/>
  <text x="48" y="144" class="ttl">① Block tile：原始 Q/K/V 的 shape → tile 怎么切（D=128：BLOCK_M × BLOCK_N = 128 × 64）</text>
  <text x="48" y="168" class="txt">原始输入：Q [B, H, L, D]　K、V 同形（TLX 按 4D [B,H,S,D] 传参）</text>
  <text x="48" y="192" class="sm">前面一层 = 一个 (b, h)；后面的阴影 = 其余 B × H 层　·　示意 L = 512 = 4 × 128 = 8 × 64</text>
  <g fill="#ffffff" stroke-width="0.9">
    <rect x="79" y="251" width="40" height="160" stroke="#2563eb"/><rect x="76" y="254" width="40" height="160" stroke="#0d9488"/><rect x="73" y="257" width="40" height="160" stroke="#0d9488"/>
    <rect x="151" y="311" width="160" height="40" stroke="#2563eb"/><rect x="148" y="314" width="160" height="40" stroke="#0d9488"/><rect x="145" y="317" width="160" height="40" stroke="#0d9488"/>
    <rect x="341" y="251" width="160" height="160" stroke="#2563eb"/><rect x="338" y="254" width="160" height="160" stroke="#0d9488"/><rect x="335" y="257" width="160" height="160" stroke="#0d9488"/>
    <rect x="691" y="251" width="40" height="160" stroke="#2563eb"/><rect x="688" y="254" width="40" height="160" stroke="#0d9488"/><rect x="685" y="257" width="40" height="160" stroke="#0d9488"/>
    <rect x="759" y="251" width="40" height="160" stroke="#2563eb"/><rect x="756" y="254" width="40" height="160" stroke="#0d9488"/><rect x="753" y="257" width="40" height="160" stroke="#0d9488"/>
  </g>
  <rect x="70" y="260" width="40" height="160" fill="#fef3c7" stroke="#b45309"/>
  <g stroke="#b45309" stroke-width="0.9" stroke-dasharray="3,2">
    <line x1="70" y1="300" x2="110" y2="300"/><line x1="70" y1="340" x2="110" y2="340"/><line x1="70" y1="380" x2="110" y2="380"/>
  </g>
  <rect x="70" y="300" width="40" height="40" fill="#fde68a" stroke="#b45309" stroke-width="2.5"/>
  <text x="90" y="218" font-size="20" font-weight="700" fill="#b45309" text-anchor="middle">Q</text>
  <text x="90" y="322" font-size="9" font-weight="700" fill="#0f172a" text-anchor="middle">m1</text>
  <text x="90" y="334" font-size="8" fill="#92400e" text-anchor="middle">128×D</text>
  <path d="M64,260 L58,260 L58,420 L64,420" fill="none" stroke="#ea580c" stroke-width="1.8"/>
  <text x="50" y="345" font-size="13" font-weight="700" fill="#ea580c" text-anchor="middle">L</text>
  <path d="M70,244 L70,238 L110,238 L110,244" fill="none" stroke="#7c3aed" stroke-width="1.8"/>
  <text x="90" y="233" font-size="13" font-weight="700" fill="#7c3aed" text-anchor="middle">D</text>
  <text x="116" y="248" font-size="11" font-weight="700"><tspan fill="#2563eb">B</tspan><tspan fill="#334155">×</tspan><tspan fill="#0d9488">H</tspan></text>
  <text x="126" y="346" font-size="18" fill="#334155" text-anchor="middle">×</text>
  <rect x="142" y="320" width="160" height="40" fill="#f3e8ff" stroke="#7c3aed"/>
  <g stroke="#7c3aed" stroke-width="0.9" stroke-dasharray="3,2"><line x1="162" y1="320" x2="162" y2="360"/><line x1="182" y1="320" x2="182" y2="360"/><line x1="202" y1="320" x2="202" y2="360"/><line x1="222" y1="320" x2="222" y2="360"/><line x1="242" y1="320" x2="242" y2="360"/><line x1="262" y1="320" x2="262" y2="360"/><line x1="282" y1="320" x2="282" y2="360"/></g>
  <rect x="162" y="320" width="20" height="40" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="222" y="218" font-size="20" font-weight="700" fill="#7c3aed" text-anchor="middle">Kᵀ</text>
  <text x="172" y="345" font-size="9" font-weight="700" fill="#0f172a" text-anchor="middle">j</text>
  <path d="M142,306 L142,300 L302,300 L302,306" fill="none" stroke="#ea580c" stroke-width="1.8"/>
  <text x="222" y="295" font-size="13" font-weight="700" fill="#ea580c" text-anchor="middle">L</text>
  <text x="222" y="376" class="sm" text-anchor="middle">D × L　(j 块 = D × 64)</text>
  <text x="317" y="346" font-size="18" fill="#334155" text-anchor="middle">=</text>
  <text x="412" y="218" font-size="20" font-weight="700" fill="#dc2626" text-anchor="middle">S / P</text><text x="412" y="242" font-size="10" text-anchor="middle" fill="#991b1b">S → softmax → P</text>
  <rect x="332" y="260" width="160" height="160" fill="#fff1f2" stroke="#dc2626"/>
  <rect x="332" y="300" width="160" height="40" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>
  <g stroke="#dc2626" stroke-width="0.7" stroke-dasharray="2,2">
    <line x1="352" y1="260" x2="352" y2="420"/><line x1="372" y1="260" x2="372" y2="420"/><line x1="392" y1="260" x2="392" y2="420"/><line x1="412" y1="260" x2="412" y2="420"/><line x1="432" y1="260" x2="432" y2="420"/><line x1="452" y1="260" x2="452" y2="420"/><line x1="472" y1="260" x2="472" y2="420"/>
    <line x1="332" y1="300" x2="492" y2="300"/><line x1="332" y1="340" x2="492" y2="340"/><line x1="332" y1="380" x2="492" y2="380"/>
  </g>
  <rect x="352" y="300" width="20" height="40" fill="#f87171" stroke="#991b1b" stroke-width="2"/>
  <rect x="352" y="260" width="20" height="160" fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="4,3"/>
  <g font-size="10">
    <text x="416" y="285" fill="#475569">m0 → WG 0</text>
    <text x="416" y="325" fill="#991b1b" font-weight="700">m1 → WG 1</text>
    <text x="416" y="365" fill="#475569">m2 → WG 2</text>
  </g>
  <g stroke-width="1.4" marker-end="url(#arr)">
    <line x1="338" y1="280" x2="486" y2="280" stroke="#94a3b8"/>
    <line x1="338" y1="320" x2="486" y2="320" stroke="#ea580c"/>
    <line x1="338" y1="360" x2="486" y2="360" stroke="#94a3b8"/>
  </g>
  <text x="502" y="272" font-size="12" font-weight="700"><tspan fill="#2563eb">B</tspan><tspan fill="#334155"> × </tspan><tspan fill="#0d9488">H</tspan><tspan fill="#334155"> 层</tspan></text>
  <text x="508" y="292" font-size="11" font-weight="700" fill="#991b1b">每一行 m = 一个 workgroup</text>
  <text x="508" y="308" class="sm">workgroup 内沿 j = 0 → n-1</text>
  <text x="508" y="324" class="sm">依次做，累加 O / m / l</text>
  <text x="508" y="350" font-size="11" font-weight="700" fill="#7c3aed">第 j 列 (K_j、V_j)</text>
  <text x="508" y="366" class="sm">被同一 (b, h) 的所有</text>
  <text x="508" y="382" class="sm">workgroup 各读一次 (只读共享)</text>
  <text x="508" y="406" class="sm">n = 512/64 = 8 次循环</text>
  <text x="666" y="346" font-size="18" fill="#334155" text-anchor="middle">×</text>
  <rect x="682" y="260" width="40" height="160" fill="#f3e8ff" stroke="#7c3aed"/>
  <g stroke="#7c3aed" stroke-width="0.9" stroke-dasharray="3,2"><line x1="682" y1="300" x2="722" y2="300"/><line x1="682" y1="340" x2="722" y2="340"/><line x1="682" y1="380" x2="722" y2="380"/></g>
  <rect x="682" y="280" width="40" height="20" fill="#c4b5fd" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="702" y="218" font-size="20" font-weight="700" fill="#7c3aed" text-anchor="middle">V</text>
  <text x="702" y="294" font-size="9" font-weight="700" fill="#0f172a" text-anchor="middle">j</text>
  <text x="736" y="346" font-size="18" fill="#334155" text-anchor="middle">=</text>
  <rect x="750" y="260" width="40" height="160" fill="#f0fdf4" stroke="#16a34a"/>
  <g stroke="#16a34a" stroke-width="0.9" stroke-dasharray="3,2">
    <line x1="750" y1="300" x2="790" y2="300"/><line x1="750" y1="340" x2="790" y2="340"/><line x1="750" y1="380" x2="790" y2="380"/>
  </g>
  <rect x="750" y="300" width="40" height="40" fill="#bbf7d0" stroke="#16a34a" stroke-width="2.5"/>
  <text x="770" y="218" font-size="20" font-weight="700" fill="#16a34a" text-anchor="middle">O</text>
  <text x="770" y="322" font-size="9" font-weight="700" fill="#0f172a" text-anchor="middle">m1</text>
  <text x="770" y="334" font-size="8" fill="#166534" text-anchor="middle">128×D</text>
  <path d="M804,300 L810,300 L810,340 L804,340" fill="none" stroke="#475569" stroke-width="1.8"/>
  <text x="814" y="322" font-size="11" font-weight="700" fill="#475569">128</text>
  <text x="814" y="335" font-size="10" fill="#475569">行</text>
  <g class="sm" text-anchor="middle">
    <text x="90" y="438">L × D</text>
    <text x="412" y="438">L × L (从不整块存在)</text>
    <text x="412" y="454">深红 = 128 × 64 (一次迭代)</text>
    <text x="702" y="438">L × D</text>
    <text x="702" y="454">j 块 = 64 × D</text>
    <text x="770" y="438">L × D</text>
  </g>
  <text x="60" y="472" class="sm">→ program_id 经 XCD remap 上 CU：连续 tile 落同一 XCD（NUM_XCDS=8 只在大 batch 开，其余为 1）</text>
  <text x="60" y="490" class="sm">→ BN=64：双缓冲、D=256 时 K/V 占 128KB LDS（上限 160KB）；BM / wave 数按 shape 和实测选择</text>
  <rect class="note" x="925" y="118" width="325" height="386"/>
  <text x="940" y="144" class="ttl" font-size="15">B / H / L / D 决定什么</text>
  <g class="sm" font-weight="700" fill="#0f172a"><text x="940" y="168">维度</text><text x="990" y="168">决定什么</text></g>
  <line x1="936" y1="175" x2="1240" y2="175" stroke="#e2e8f0"/>
  <rect class="chipB" x="940" y="184" width="34" height="18" rx="3"/><text x="957" y="197" class="chipT" text-anchor="middle">B</text>
  <text x="990" y="196" class="txt" font-size="12">层数 → tile 个数 × B</text>
  <rect class="chipH" x="940" y="212" width="34" height="18" rx="3"/><text x="957" y="225" class="chipT" text-anchor="middle">H</text>
  <text x="990" y="224" class="txt" font-size="12">层数 → tile 个数 × H</text>
  <rect class="chipL" x="940" y="240" width="34" height="18" rx="3"/><text x="957" y="253" class="chipT" text-anchor="middle">L</text>
  <text x="990" y="252" class="txt" font-size="12">Q 切 ceil(L/128) 个 tile → 个数</text>
  <text x="990" y="270" class="txt" font-size="12">K/V 切 L/64 块 → 循环次数 n</text>
  <rect class="chipD" x="940" y="282" width="34" height="18" rx="3"/><text x="957" y="295" class="chipT" text-anchor="middle">D</text>
  <text x="990" y="294" class="txt" font-size="12">完整 D 参与计算；BM 按 shape 选</text>
  <text x="990" y="312" class="mono" font-size="11">BM 可为 128 / 256</text>
  <text x="990" y="328" class="mono" font-size="11">num_warps 可为 4 / 8</text>
  <text x="990" y="344" class="sm">(BLOCK_N = 64)</text>
  <text x="940" y="368" class="sm">tile 总数 = B×H×ceil(L/BM)</text>
  <text x="940" y="386" class="sm">L 唯一两层都影响（个数+循环）</text>
  <text x="940" y="410" class="sm">对比 FA4：tile 固定 256×128，</text>
  <text x="940" y="428" class="sm">常驻 CTA 循环领 tile（无 XCD）</text>
<rect class="lvl" x="30" y="512" width="880" height="940"/>
<text x="48" y="538" font-size="17" fill="#0f172a" font-weight="700">② Workgroup 内：按 wave 画执行流，协作处理六个 K/V tile（j=0…5）</text>
<text x="48" y="560" font-size="11" fill="#334155">只展示 wave 0 / 1（实际 4 或 8 个）；Kj[w] / Vj[w] 表示 wave w 发起的搬运部分，不是数组行号。</text>
<text x="60" y="586" font-size="13" fill="#0f172a" font-weight="700">②a 整个 workgroup：固定一个 Q 行块，沿 KV0 → KV5 循环</text>
<rect x="60" y="617" width="80" height="96" rx="3" fill="#bfdbfe" stroke="#2563eb"/>
<text x="100.0" y="669.55" font-size="13" fill="#334155" text-anchor="middle">Q 行块</text>
<text x="100" y="699" font-size="10" fill="#334155" text-anchor="middle">BM × D</text>
<line x1="144" y1="665" x2="197" y2="665" stroke="#64748b" stroke-width="1.3" marker-end="url(#arr)"/>
<text x="247" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 0 轮</text>
<rect x="205" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="247.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K0 · 64×D</text>
<rect x="205" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="247.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V0 · 64×D</text>
<rect x="200" y="617" width="94" height="96" rx="5" fill="none" stroke="#ea580c" stroke-width="2"/>
<text x="359" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 1 轮</text>
<rect x="317" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="359.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K1 · 64×D</text>
<rect x="317" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="359.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V1 · 64×D</text>
<text x="471" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 2 轮</text>
<rect x="429" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="471.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K2 · 64×D</text>
<rect x="429" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="471.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V2 · 64×D</text>
<text x="583" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 3 轮</text>
<rect x="541" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="583.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K3 · 64×D</text>
<rect x="541" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="583.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V3 · 64×D</text>
<text x="695" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 4 轮</text>
<rect x="653" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="695.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K4 · 64×D</text>
<rect x="653" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="695.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V4 · 64×D</text>
<text x="807" y="613" font-size="11" fill="#334155" text-anchor="middle" font-weight="700">第 5 轮</text>
<rect x="765" y="622" width="84" height="39" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="807.0" y="645.35" font-size="11" fill="#334155" text-anchor="middle">K5 · 64×D</text>
<rect x="765" y="669" width="84" height="39" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="807.0" y="692.35" font-size="11" fill="#334155" text-anchor="middle">V5 · 64×D</text>
<line x1="210" y1="729" x2="851" y2="729" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="530" y="750" font-size="11" fill="#334155" text-anchor="middle">每轮复用同一个 Q block tile；第 j 轮使用第 j 个 K/V block tile。</text>
<g transform="translate(0,190)">
<text x="60" y="586" font-size="13" fill="#0f172a" font-weight="700">②b 放大第 0 轮：Q / K0 / V0 的 wave 分片 → 协作计算 S0</text>
<text x="60" y="600" font-size="10" fill="#334155">计算分块示意（非实际编译布局）：Kᵀ 沿 N 分列，V0 沿 D 分列。</text>
<text x="115.0" y="617" font-size="12" fill="#2563eb" text-anchor="middle" font-weight="700">Q · 寄存器</text>
<rect x="60" y="626" width="110" height="25" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="115.0" y="642.0" font-size="10" fill="#334155" text-anchor="middle">wave 0 的片段</text>
<rect x="60" y="651" width="110" height="25" rx="3" fill="#ccfbf1" stroke="#0d9488"/>
<text x="115.0" y="667.0" font-size="10" fill="#334155" text-anchor="middle">wave 1 的片段</text>
<rect x="60" y="676" width="110" height="25" rx="3" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="115.0" y="692.0" font-size="10" fill="#334155" text-anchor="middle">其他 wave 片段</text>
<rect x="60" y="701" width="110" height="25" rx="3" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="115.0" y="717.0" font-size="10" fill="#334155" text-anchor="middle">其他 wave 片段</text>
<rect x="60" y="626" width="110" height="100" fill="none" stroke="#2563eb" stroke-width="1.5"/>
<text x="115.0" y="744" font-size="11" fill="#334155" text-anchor="middle">BM × D</text>
<text x="271.0" y="617" font-size="12" fill="#16a34a" text-anchor="middle" font-weight="700">K0ᵀ · 计算视图</text>
<rect x="196.0" y="626" width="37.5" height="100" fill="#dbeafe" stroke="#2563eb"/>
<text x="214.75" y="662" font-size="9" fill="#334155" text-anchor="middle">wave 0</text>
<text x="214.75" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="233.5" y="626" width="37.5" height="100" fill="#ccfbf1" stroke="#0d9488"/>
<text x="252.25" y="662" font-size="9" fill="#334155" text-anchor="middle">wave 1</text>
<text x="252.25" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="271.0" y="626" width="37.5" height="100" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="289.75" y="662" font-size="9" fill="#334155" text-anchor="middle">其他</text>
<text x="289.75" y="678" font-size="9" fill="#334155" text-anchor="middle">wave</text>
<text x="289.75" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="308.5" y="626" width="37.5" height="100" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="327.25" y="662" font-size="9" fill="#334155" text-anchor="middle">其他</text>
<text x="327.25" y="678" font-size="9" fill="#334155" text-anchor="middle">wave</text>
<text x="327.25" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="196" y="626" width="150" height="100" fill="none" stroke="#16a34a" stroke-width="1.5"/>
<text x="271.0" y="744" font-size="11" fill="#334155" text-anchor="middle">D × 64</text>
<text x="441.0" y="617" font-size="12" fill="#7c3aed" text-anchor="middle" font-weight="700">V0 · PV 计算</text>
<rect x="366.0" y="626" width="37.5" height="100" fill="#dbeafe" stroke="#2563eb"/>
<text x="384.75" y="662" font-size="9" fill="#334155" text-anchor="middle">wave 0</text>
<text x="384.75" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="403.5" y="626" width="37.5" height="100" fill="#ccfbf1" stroke="#0d9488"/>
<text x="422.25" y="662" font-size="9" fill="#334155" text-anchor="middle">wave 1</text>
<text x="422.25" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="441.0" y="626" width="37.5" height="100" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="459.75" y="662" font-size="9" fill="#334155" text-anchor="middle">其他</text>
<text x="459.75" y="678" font-size="9" fill="#334155" text-anchor="middle">wave</text>
<text x="459.75" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="478.5" y="626" width="37.5" height="100" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="497.25" y="662" font-size="9" fill="#334155" text-anchor="middle">其他</text>
<text x="497.25" y="678" font-size="9" fill="#334155" text-anchor="middle">wave</text>
<text x="497.25" y="697" font-size="9" fill="#334155" text-anchor="middle">的片段</text>
<rect x="366" y="626" width="150" height="100" fill="none" stroke="#7c3aed" stroke-width="1.5"/>
<text x="441.0" y="744" font-size="11" fill="#334155" text-anchor="middle">64 × D</text>
<text x="797.0" y="617" font-size="12" fill="#dc2626" text-anchor="middle" font-weight="700">S0 = Q × K0ᵀ</text>
<rect x="722" y="626" width="150" height="25" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="797.0" y="642.0" font-size="10" fill="#334155" text-anchor="middle">wave 0 的片段</text>
<rect x="722" y="651" width="150" height="25" rx="3" fill="#ccfbf1" stroke="#0d9488"/>
<text x="797.0" y="667.0" font-size="10" fill="#334155" text-anchor="middle">wave 1 的片段</text>
<rect x="722" y="676" width="150" height="25" rx="3" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="797.0" y="692.0" font-size="10" fill="#334155" text-anchor="middle">其他 wave 片段</text>
<rect x="722" y="701" width="150" height="25" rx="3" fill="#f1f5f9" stroke="#94a3b8"/>
<text x="797.0" y="717.0" font-size="10" fill="#334155" text-anchor="middle">其他 wave 片段</text>
<rect x="722" y="626" width="150" height="100" fill="none" stroke="#dc2626" stroke-width="1.5"/>
<text x="797.0" y="744" font-size="11" fill="#334155" text-anchor="middle">BM × 64</text>
<rect x="528" y="622" width="150" height="104" rx="6" fill="#ffffff" stroke="#64748b"/>
<text x="603" y="641" font-size="13" fill="#0f172a" text-anchor="middle" font-weight="700">共享 LDS</text>
<rect x="536" y="649" width="134" height="31" rx="3" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="603.0" y="667.65" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="541" y="668" font-size="8" fill="#334155">slot 0</text>
<rect x="574" y="654" width="43" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="595.5" y="667.3" font-size="8" fill="#334155" text-anchor="middle">K[0]</text>
<rect x="621" y="654" width="43" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="642.5" y="667.3" font-size="8" fill="#334155" text-anchor="middle">V[0]</text>
<rect x="536" y="685" width="134" height="31" rx="3" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="603.0" y="703.65" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="541" y="704" font-size="8" fill="#334155">slot 1</text>
<rect x="574" y="690" width="43" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="595.5" y="703.3" font-size="8" fill="#334155" text-anchor="middle">K[1]</text>
<rect x="621" y="690" width="43" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="642.5" y="703.3" font-size="8" fill="#334155" text-anchor="middle">V[1]</text>
<path d="M518,674 L524,674 L524,664 L534,664" fill="none" stroke="#64748b" stroke-width="1.2" marker-end="url(#arr)"/>
<path d="M271,748 L271,762 L522,762 L522,664 L534,664" fill="none" stroke="#64748b" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="672" y1="664" x2="714" y2="664" stroke="#64748b" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="698" y="640" font-size="9" fill="#334155" text-anchor="middle">配合 Q</text>
<text x="698" y="653" font-size="9" fill="#334155" text-anchor="middle">算 S0</text>
</g>
<g transform="translate(0,450)">
<text x="48" y="582" font-size="11" fill="#334155">Q 先读一次；每轮：等当前 K/V → 发下一块 K/V → S=QKᵀ → M/C（softmax、修正）→ PV（累积 O）</text>
<text x="214" y="607" font-size="10" fill="#334155" text-anchor="middle" font-weight="700">启动</text>
<text x="300" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 0 轮</text>
<text x="300" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 0</text>
<line x1="247" y1="614" x2="247" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<text x="406" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 1 轮</text>
<text x="406" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 1</text>
<line x1="353" y1="614" x2="353" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<text x="512" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 2 轮</text>
<text x="512" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 0</text>
<line x1="459" y1="614" x2="459" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<text x="618" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 3 轮</text>
<text x="618" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 1</text>
<line x1="565" y1="614" x2="565" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<text x="724" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 4 轮</text>
<text x="724" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 0</text>
<line x1="671" y1="614" x2="671" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<text x="830" y="600" font-size="10" fill="#b45309" text-anchor="middle" font-weight="700">第 5 轮</text>
<text x="830" y="613" font-size="10" fill="#334155" text-anchor="middle">读 slot 1</text>
<line x1="777" y1="614" x2="777" y2="822" stroke="#e2e8f0" stroke-dasharray="3,3"/>
<rect x="40" y="622" width="5" height="82" rx="2" fill="#2563eb"/>
<text x="54" y="666" font-size="14" fill="#2563eb" font-weight="700">wave 0</text>
<text x="54" y="685" font-size="10" fill="#334155">分摊同一 tile</text>
<rect x="188" y="622" width="50" height="20" rx="3" fill="#bfdbfe" stroke="#2563eb"/>
<text x="213.0" y="635.15" font-size="9" fill="#334155" text-anchor="middle">读 Q</text>
<rect x="188" y="646" width="50" height="18" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="213.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K0[0]</text>
<rect x="188" y="666" width="50" height="18" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="213.0" y="677.8" font-size="8" fill="#334155" text-anchor="middle">发 V0[0]</text>
<text x="213" y="700" font-size="9" fill="#334155" text-anchor="middle">写 slot 0</text>
<rect x="250" y="645" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="274.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K1[0]</text>
<rect x="302" y="645" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="326.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 V1[0]</text>
<path d="M251,682 L255,677 L259,682 L255,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="263" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="273.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S0</text>
<text x="273.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="288" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="305.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M0/C0</text>
<text x="305.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="326" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="338.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV0</text>
<text x="338.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="356" y="645" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="380.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K2[0]</text>
<rect x="408" y="645" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="432.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 V2[0]</text>
<path d="M357,682 L361,677 L365,682 L361,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="369" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="379.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S1</text>
<text x="379.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="394" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="411.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M1/C1</text>
<text x="411.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="432" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="444.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV1</text>
<text x="444.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="462" y="645" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="486.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K3[0]</text>
<rect x="514" y="645" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="538.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 V3[0]</text>
<path d="M463,682 L467,677 L471,682 L467,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="475" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="485.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S2</text>
<text x="485.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="500" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="517.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M2/C2</text>
<text x="517.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="538" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="550.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV2</text>
<text x="550.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="568" y="645" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="592.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K4[0]</text>
<rect x="620" y="645" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="644.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 V4[0]</text>
<path d="M569,682 L573,677 L577,682 L573,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="581" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="591.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S3</text>
<text x="591.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="606" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="623.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M3/C3</text>
<text x="623.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="644" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="656.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV3</text>
<text x="656.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="674" y="645" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="698.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 K5[0]</text>
<rect x="726" y="645" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="750.0" y="657.8" font-size="8" fill="#334155" text-anchor="middle">发 V5[0]</text>
<path d="M675,682 L679,677 L683,682 L679,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="687" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="697.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S4</text>
<text x="697.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="712" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="729.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M4/C4</text>
<text x="729.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="750" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="762.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV4</text>
<text x="762.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="780" y="645" width="100" height="20" rx="3" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="830.0" y="658.15" font-size="9" fill="#334155" text-anchor="middle">末块：不再预取</text>
<path d="M781,682 L785,677 L789,682 L785,687 z" fill="#ffffff" stroke="#64748b"/>
<rect x="793" y="670" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="803.5" y="680" font-size="8" fill="#334155" text-anchor="middle">S5</text>
<text x="803.5" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="818" y="670" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="835.0" y="680" font-size="8" fill="#334155" text-anchor="middle">M5/C5</text>
<text x="835.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="856" y="670" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="868.0" y="680" font-size="8" fill="#334155" text-anchor="middle">PV5</text>
<text x="868.0" y="690" font-size="8" fill="#334155" text-anchor="middle">[0]</text>
<rect x="40" y="726" width="5" height="82" rx="2" fill="#0d9488"/>
<text x="54" y="770" font-size="14" fill="#0d9488" font-weight="700">wave 1</text>
<text x="54" y="789" font-size="10" fill="#334155">分摊同一 tile</text>
<rect x="188" y="726" width="50" height="20" rx="3" fill="#bfdbfe" stroke="#2563eb"/>
<text x="213.0" y="739.15" font-size="9" fill="#334155" text-anchor="middle">读 Q</text>
<rect x="188" y="750" width="50" height="18" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="213.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K0[1]</text>
<rect x="188" y="770" width="50" height="18" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="213.0" y="781.8" font-size="8" fill="#334155" text-anchor="middle">发 V0[1]</text>
<text x="213" y="804" font-size="9" fill="#334155" text-anchor="middle">写 slot 0</text>
<rect x="250" y="749" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="274.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K1[1]</text>
<rect x="302" y="749" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="326.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 V1[1]</text>
<path d="M251,786 L255,781 L259,786 L255,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="263" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="273.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S0</text>
<text x="273.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="288" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="305.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M0/C0</text>
<text x="305.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="326" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="338.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV0</text>
<text x="338.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="356" y="749" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="380.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K2[1]</text>
<rect x="408" y="749" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="432.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 V2[1]</text>
<path d="M357,786 L361,781 L365,786 L361,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="369" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="379.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S1</text>
<text x="379.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="394" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="411.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M1/C1</text>
<text x="411.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="432" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="444.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV1</text>
<text x="444.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="462" y="749" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="486.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K3[1]</text>
<rect x="514" y="749" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="538.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 V3[1]</text>
<path d="M463,786 L467,781 L471,786 L467,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="475" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="485.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S2</text>
<text x="485.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="500" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="517.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M2/C2</text>
<text x="517.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="538" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="550.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV2</text>
<text x="550.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="568" y="749" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="592.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K4[1]</text>
<rect x="620" y="749" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="644.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 V4[1]</text>
<path d="M569,786 L573,781 L577,786 L573,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="581" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="591.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S3</text>
<text x="591.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="606" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="623.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M3/C3</text>
<text x="623.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="644" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="656.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV3</text>
<text x="656.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="674" y="749" width="48" height="20" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="698.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 K5[1]</text>
<rect x="726" y="749" width="48" height="20" rx="3" fill="#e9d5ff" stroke="#7c3aed"/>
<text x="750.0" y="761.8" font-size="8" fill="#334155" text-anchor="middle">发 V5[1]</text>
<path d="M675,786 L679,781 L683,786 L679,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="687" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="697.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S4</text>
<text x="697.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="712" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="729.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M4/C4</text>
<text x="729.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="750" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="762.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV4</text>
<text x="762.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="780" y="749" width="100" height="20" rx="3" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="830.0" y="762.15" font-size="9" fill="#334155" text-anchor="middle">末块：不再预取</text>
<path d="M781,786 L785,781 L789,786 L785,791 z" fill="#ffffff" stroke="#64748b"/>
<rect x="793" y="774" width="21" height="24" rx="3" fill="#fecaca" stroke="#dc2626"/>
<text x="803.5" y="784" font-size="8" fill="#334155" text-anchor="middle">S5</text>
<text x="803.5" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="818" y="774" width="34" height="24" rx="3" fill="#fde68a" stroke="#b45309"/>
<text x="835.0" y="784" font-size="8" fill="#334155" text-anchor="middle">M5/C5</text>
<text x="835.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<rect x="856" y="774" width="24" height="24" rx="3" fill="#bbf7d0" stroke="#16a34a"/>
<text x="868.0" y="784" font-size="8" fill="#334155" text-anchor="middle">PV5</text>
<text x="868.0" y="794" font-size="8" fill="#334155" text-anchor="middle">[1]</text>
<text x="54" y="856" font-size="12" fill="#0f172a" font-weight="700">LDS slot 0</text>
<text x="54" y="874" font-size="10" fill="#334155">同一个物理槽</text>
<rect x="180" y="831" width="710" height="58" rx="6" fill="#ffffff" stroke="#64748b" stroke-width="1.4"/>
<rect x="250" y="835" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="300.0" y="863.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="300" y="848" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV0</text>
<rect x="253" y="856" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="275.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">K[0]</text>
<rect x="302" y="856" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="324.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">V[0]</text>
<rect x="356" y="835" width="100" height="50" rx="3" fill="#eff6ff" stroke="#2563eb"/>
<text x="406.0" y="863.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="406" y="848" font-size="9" fill="#1d4ed8" text-anchor="middle" font-weight="700">写入→就绪 · KV2</text>
<rect x="359" y="852" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="381.5" y="860.95" font-size="7" fill="#334155" text-anchor="middle">K2[0]</text>
<rect x="359" y="867" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="381.5" y="875.95" font-size="7" fill="#334155" text-anchor="middle">V2[0]</text>
<rect x="408" y="852" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="430.5" y="860.95" font-size="7" fill="#334155" text-anchor="middle">K2[1]</text>
<rect x="408" y="867" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="430.5" y="875.95" font-size="7" fill="#334155" text-anchor="middle">V2[1]</text>
<rect x="462" y="835" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="512.0" y="863.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="512" y="848" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV2</text>
<rect x="465" y="856" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="487.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">K[2]</text>
<rect x="514" y="856" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="536.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">V[2]</text>
<rect x="568" y="835" width="100" height="50" rx="3" fill="#eff6ff" stroke="#2563eb"/>
<text x="618.0" y="863.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="618" y="848" font-size="9" fill="#1d4ed8" text-anchor="middle" font-weight="700">写入→就绪 · KV4</text>
<rect x="571" y="852" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="593.5" y="860.95" font-size="7" fill="#334155" text-anchor="middle">K4[0]</text>
<rect x="571" y="867" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="593.5" y="875.95" font-size="7" fill="#334155" text-anchor="middle">V4[0]</text>
<rect x="620" y="852" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="642.5" y="860.95" font-size="7" fill="#334155" text-anchor="middle">K4[1]</text>
<rect x="620" y="867" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="642.5" y="875.95" font-size="7" fill="#334155" text-anchor="middle">V4[1]</text>
<rect x="674" y="835" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="724.0" y="863.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="724" y="848" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV4</text>
<rect x="677" y="856" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="699.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">K[4]</text>
<rect x="726" y="856" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="748.5" y="869.65" font-size="9" fill="#334155" text-anchor="middle">V[4]</text>
<rect x="780" y="835" width="100" height="50" rx="3" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="830.0" y="863.5" font-size="10" fill="#334155" text-anchor="middle">空闲</text>
<text x="54" y="924" font-size="12" fill="#0f172a" font-weight="700">LDS slot 1</text>
<text x="54" y="942" font-size="10" fill="#334155">同一个物理槽</text>
<rect x="180" y="899" width="710" height="58" rx="6" fill="#ffffff" stroke="#64748b" stroke-width="1.4"/>
<rect x="250" y="903" width="100" height="50" rx="3" fill="#eff6ff" stroke="#2563eb"/>
<text x="300.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="300" y="916" font-size="9" fill="#1d4ed8" text-anchor="middle" font-weight="700">写入→就绪 · KV1</text>
<rect x="253" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="275.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K1[0]</text>
<rect x="253" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="275.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V1[0]</text>
<rect x="302" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="324.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K1[1]</text>
<rect x="302" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="324.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V1[1]</text>
<rect x="356" y="903" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="406.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="406" y="916" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV1</text>
<rect x="359" y="924" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="381.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">K[1]</text>
<rect x="408" y="924" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="430.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">V[1]</text>
<rect x="462" y="903" width="100" height="50" rx="3" fill="#eff6ff" stroke="#2563eb"/>
<text x="512.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="512" y="916" font-size="9" fill="#1d4ed8" text-anchor="middle" font-weight="700">写入→就绪 · KV3</text>
<rect x="465" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="487.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K3[0]</text>
<rect x="465" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="487.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V3[0]</text>
<rect x="514" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="536.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K3[1]</text>
<rect x="514" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="536.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V3[1]</text>
<rect x="568" y="903" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="618.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="618" y="916" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV3</text>
<rect x="571" y="924" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="593.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">K[3]</text>
<rect x="620" y="924" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="642.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">V[3]</text>
<rect x="674" y="903" width="100" height="50" rx="3" fill="#eff6ff" stroke="#2563eb"/>
<text x="724.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="724" y="916" font-size="9" fill="#1d4ed8" text-anchor="middle" font-weight="700">写入→就绪 · KV5</text>
<rect x="677" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="699.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K5[0]</text>
<rect x="677" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="699.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V5[0]</text>
<rect x="726" y="920" width="45" height="13" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="748.5" y="928.95" font-size="7" fill="#334155" text-anchor="middle">K5[1]</text>
<rect x="726" y="935" width="45" height="13" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="748.5" y="943.95" font-size="7" fill="#334155" text-anchor="middle">V5[1]</text>
<rect x="780" y="903" width="100" height="50" rx="3" fill="#fff7ed" stroke="#ea580c"/>
<text x="830.0" y="931.15" font-size="9" fill="#334155" text-anchor="middle"></text>
<text x="830" y="916" font-size="9" fill="#9a3412" text-anchor="middle" font-weight="700">已就绪 · KV5</text>
<rect x="783" y="924" width="45" height="21" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="805.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">K[5]</text>
<rect x="832" y="924" width="45" height="21" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="854.5" y="937.65" font-size="9" fill="#334155" text-anchor="middle">V[5]</text>
<text x="60" y="981" font-size="11" fill="#334155">每轮：当前槽已就绪，另一槽写下一块；两个槽由全部 wave 共享。◇：计算前确认就绪。</text>
</g>
<rect class="note" x="925" y="512" width="325" height="940"/>
<text x="940" y="538" font-size="16" fill="#334155" font-weight="700">当前 AMD FA 的分工</text>
<text x="940" y="578" font-size="12" fill="#334155">一个 Q 行块 → 一个 workgroup</text>
<text x="940" y="606" font-size="12" fill="#334155">4 / 8 个 wave 协作搬运和计算</text>
<text x="940" y="644" font-size="12" fill="#334155" font-weight="700">每个 wave：处理分配到的元素</text>
<text x="940" y="671" font-size="11" fill="#334155">同一批 wave 参与所有计算阶段</text>
<text x="940" y="709" font-size="12" fill="#334155" font-weight="700">双缓冲：slot = j % 2</text>
<text x="940" y="737" font-size="11" fill="#334155">j=0/2/4 → slot 0</text>
<text x="940" y="762" font-size="11" fill="#334155">j=1/3/5 → slot 1</text>
<text x="940" y="800" font-size="12" fill="#334155" font-weight="700">预取深度：下一对 K/V</text>
<text x="940" y="828" font-size="11" fill="#334155">先发下一块，再算当前块</text>
<text x="940" y="853" font-size="11" fill="#334155">就绪检查不代表固定等待时长</text>
<text x="940" y="891" font-size="12" fill="#334155" font-weight="700">对比 Blackwell FA4</text>
<text x="940" y="918" font-size="11" fill="#334155">BW 有专用搬运 / MMA / M/C warp</text>
<text x="940" y="945" font-size="11" fill="#334155">此 TLX kernel 没有按角色拆 wave</text>
<text x="940" y="976" font-size="11" fill="#334155">术语对应项集中在图底部</text>
<g transform="translate(0,680)">
<rect x="30" y="792" width="1220" height="340" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="818" font-size="17" fill="#0f172a" font-weight="700">③ Lane tiling：一个 wave 的矩阵片段 → 64 个 lane 各自的寄存器元素</text>
<text x="60" y="842" font-size="11" fill="#334155">逻辑归属示意：AMD wave 有 64 lane，Blackwell warp 有 32 lane；实际元素分配由编译布局决定。</text>
<rect x="60" y="860" width="220" height="212" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="170" y="884" font-size="12" fill="#0f172a" font-weight="700" text-anchor="middle">wave 0 使用的矩阵片段</text>
<rect x="72" y="898" width="96" height="38" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="120.0" y="920.85" font-size="11" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][0]</text>
<rect x="176" y="898" width="96" height="38" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="224.0" y="920.85" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][0]</text>
<rect x="72" y="949" width="96" height="38" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="120.0" y="971.85" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][0]</text>
<rect x="176" y="949" width="96" height="38" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="224.0" y="971.85" font-size="11" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][0]</text>
<rect x="72" y="1000" width="96" height="38" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="120.0" y="1022.85" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">P[j][0]</text>
<rect x="176" y="1000" width="96" height="38" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="224.0" y="1022.85" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][0]</text>
<line x1="286" y1="966" x2="344" y2="966" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="315" y="950" font-size="10" fill="#334155" text-anchor="middle">元素分配</text>
<rect x="350" y="860" width="440" height="212" rx="6" fill="#f8fafc" stroke="#64748b"/>
<text x="362" y="884" font-size="12" fill="#0f172a" font-weight="700">wave 0：64 个 lane（只展开 lane 0 / 1）</text>
<text x="362" y="905" font-size="11" fill="#334155">各 lane 私有寄存器 → 同一条 MFMA 指令</text>
<rect x="362" y="918" width="416" height="60" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="374" y="936" font-size="11" fill="#0f172a" font-weight="700">lane 0 · 私有寄存器</text>
<rect x="370" y="946" width="62" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="401.0" y="959.8" font-size="8" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][0][0]</text>
<rect x="437" y="946" width="62" height="22" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="468.0" y="959.8" font-size="8" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][0][0]</text>
<rect x="504" y="946" width="62" height="22" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="535.0" y="959.8" font-size="8" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][0][0]</text>
<rect x="571" y="946" width="62" height="22" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="602.0" y="959.8" font-size="8" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][0][0]</text>
<rect x="638" y="946" width="62" height="22" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="669.0" y="959.8" font-size="8" fill="#b45309" font-weight="700" text-anchor="middle">P[j][0][0]</text>
<rect x="705" y="946" width="62" height="22" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="736.0" y="959.8" font-size="8" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][0][0]</text>
<rect x="362" y="990" width="416" height="60" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="374" y="1008" font-size="11" fill="#0f172a" font-weight="700">lane 1 · 私有寄存器</text>
<rect x="370" y="1018" width="62" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="401.0" y="1031.8" font-size="8" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][0][1]</text>
<rect x="437" y="1018" width="62" height="22" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="468.0" y="1031.8" font-size="8" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][0][1]</text>
<rect x="504" y="1018" width="62" height="22" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="535.0" y="1031.8" font-size="8" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][0][1]</text>
<rect x="571" y="1018" width="62" height="22" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="602.0" y="1031.8" font-size="8" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][0][1]</text>
<rect x="638" y="1018" width="62" height="22" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="669.0" y="1031.8" font-size="8" fill="#b45309" font-weight="700" text-anchor="middle">P[j][0][1]</text>
<rect x="705" y="1018" width="62" height="22" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="736.0" y="1031.8" font-size="8" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][0][1]</text>
<rect x="840" y="860" width="380" height="212" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="1030" y="884" font-size="13" fill="#0f172a" font-weight="700" text-anchor="middle">CU 内的矩阵执行单元（阵列示意）</text>
<rect x="854" y="917" width="144" height="121" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="926" y="944" font-size="12" fill="#1d4ed8" font-weight="700" text-anchor="middle">wave 0</text>
<text x="926" y="970" font-size="13" fill="#1d4ed8" font-weight="700" text-anchor="middle">MFMA 指令</text>
<text x="926" y="997" font-size="9" fill="#334155" text-anchor="middle">64 个 lane 提供操作数</text>
<text x="926" y="1020" font-size="10" fill="#334155" text-anchor="middle">计算 QKᵀ 或 PV</text>
<path d="M778,957 L814,957 L814,948 L850,948" fill="none" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<path d="M778,1029 L814,1029 L814,1009 L850,1009" fill="none" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="1020" y="905" width="188" height="139" rx="4" fill="#f8fafc" stroke="#64748b"/>
<rect x="1030" y="916" width="168" height="47" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="1114" y="944" font-size="12" fill="#1d4ed8" font-weight="700" text-anchor="middle">矩阵执行单元 A</text>
<rect x="1030" y="981" width="168" height="47" rx="4" fill="#eff6ff" stroke="#93c5fd"/>
<text x="1114" y="1009" font-size="12" fill="#1d4ed8" font-weight="700" text-anchor="middle">矩阵执行单元 B</text>
<line x1="1001" y1="976" x2="1016" y2="976" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="1030" y="1062" font-size="11" fill="#334155" text-anchor="middle">计算结果分布回各 lane 寄存器</text>
<text x="60" y="1108" font-size="11" fill="#334155">两条 lane 只作示例；MFMA 由整个 wave 协同发出，单元阵列不表示一条 lane 对应一个单元。</text>
<g transform="translate(0,360)">
<rect x="30" y="792" width="1220" height="840" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="48" y="818" font-size="17" fill="#0f172a" font-weight="700">④ 片上存储：block tile 存在哪里 → wave / warp 使用哪些片段</text>
<text x="60" y="850" font-size="14" fill="#0f172a" font-weight="700">AMD TLX：1 workgroup · BM=128 示例</text>
<text x="660" y="850" font-size="14" fill="#0f172a" font-weight="700">Blackwell FA4：1 CTA · 256×128 示例（BF16）</text>
<rect x="60" y="869" width="560" height="55" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="887" font-size="11" fill="#334155" font-weight="700">HBM / 显存：输入与最终输出</text>
<rect x="72" y="895" width="128" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="136.0" y="910.0" font-size="11" fill="#2563eb" font-weight="700" text-anchor="middle">Q</text>
<rect x="208" y="895" width="128" height="22" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="272.0" y="910.0" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K</text>
<rect x="344" y="895" width="128" height="22" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="408.0" y="910.0" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V</text>
<rect x="480" y="895" width="128" height="22" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="544.0" y="910.0" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<rect x="660" y="869" width="560" height="55" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="672" y="887" font-size="11" fill="#334155" font-weight="700">HBM / 显存：输入与最终输出</text>
<rect x="672" y="895" width="128" height="22" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="736.0" y="910.0" font-size="11" fill="#2563eb" font-weight="700" text-anchor="middle">Q</text>
<rect x="808" y="895" width="128" height="22" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="872.0" y="910.0" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K</text>
<rect x="944" y="895" width="128" height="22" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="1008.0" y="910.0" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V</text>
<rect x="1080" y="895" width="128" height="22" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="1144.0" y="910.0" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O</text>
<rect x="60" y="957" width="560" height="114" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="978" font-size="12" fill="#0f172a" font-weight="700">共享 LDS：K/V block tile，双缓冲（第 0 轮示例）</text>
<rect x="72" y="988" width="264" height="71" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="204" y="1005" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">slot 0 · 当前块</text>
<rect x="80" y="1012" width="120" height="39" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="140.0" y="1029" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K[0]</text>
<text x="140.0" y="1046" font-size="10" fill="#334155" text-anchor="middle">64 × D</text>
<rect x="208" y="1012" width="120" height="39" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="268.0" y="1029" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V[0]</text>
<text x="268.0" y="1046" font-size="10" fill="#334155" text-anchor="middle">64 × D</text>
<rect x="344" y="988" width="264" height="71" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="476" y="1005" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">slot 1 · 下一块</text>
<rect x="352" y="1012" width="120" height="39" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="412.0" y="1029" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K[1]</text>
<text x="412.0" y="1046" font-size="10" fill="#334155" text-anchor="middle">64 × D</text>
<rect x="480" y="1012" width="120" height="39" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="540.0" y="1029" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V[1]</text>
<text x="540.0" y="1046" font-size="10" fill="#334155" text-anchor="middle">64 × D</text>
<line x1="340" y1="924" x2="340" y2="952" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="350" y="944" font-size="11" fill="#64748b">K/V：async_load</text>
<path d="M136,924 L56,924 L56,1211 L90,1211" fill="none" stroke="#2563eb" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="72" y="1095" font-size="11" fill="#2563eb">Q：直接 tl.load</text>
<line x1="340" y1="1071" x2="340" y2="1102" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="350" y="1095" font-size="11" fill="#64748b">LDS 读取 → K/V 寄存器片段</text>
<rect x="60" y="1106" width="560" height="426" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="72" y="1128" font-size="12" fill="#0f172a" font-weight="700">寄存器：每个 lane 私有；Q/S/P/O block tile 分散存储</text>
<text x="72" y="1146" font-size="10" fill="#334155">Q/O：BM×D　·　S/P：BM×64　·　K/V：从 LDS 读出的工作元素</text>
<rect x="72" y="1158" width="536" height="164" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="84" y="1174" font-size="11" fill="#334155" font-weight="700">wave 0：64 个 lane（只展开 0 / 1）</text>
<rect x="84" y="1182" width="248" height="132" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="208" y="1198" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">lane 0 · 私有寄存器</text>
<rect x="92" y="1206" width="232" height="14" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="208" y="1216" font-size="9" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][0][0]</text>
<rect x="92" y="1222" width="232" height="14" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="208" y="1232" font-size="9" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][0][0]</text>
<rect x="92" y="1238" width="232" height="14" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="208" y="1248" font-size="9" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][0][0]</text>
<rect x="92" y="1254" width="232" height="14" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="208" y="1264" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][0][0]</text>
<rect x="92" y="1270" width="232" height="14" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="208" y="1280" font-size="9" fill="#b45309" font-weight="700" text-anchor="middle">P[j][0][0]</text>
<rect x="92" y="1286" width="232" height="14" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="208" y="1296" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][0][0]</text>
<rect x="340" y="1182" width="248" height="132" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="464" y="1198" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">lane 1 · 私有寄存器</text>
<rect x="348" y="1206" width="232" height="14" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="464" y="1216" font-size="9" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][0][1]</text>
<rect x="348" y="1222" width="232" height="14" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="464" y="1232" font-size="9" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][0][1]</text>
<rect x="348" y="1238" width="232" height="14" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="464" y="1248" font-size="9" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][0][1]</text>
<rect x="348" y="1254" width="232" height="14" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="464" y="1264" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][0][1]</text>
<rect x="348" y="1270" width="232" height="14" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="464" y="1280" font-size="9" fill="#b45309" font-weight="700" text-anchor="middle">P[j][0][1]</text>
<rect x="348" y="1286" width="232" height="14" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="464" y="1296" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][0][1]</text>
<rect x="72" y="1334" width="536" height="164" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="84" y="1350" font-size="11" fill="#334155" font-weight="700">wave 1：64 个 lane（只展开 0 / 1）</text>
<rect x="84" y="1358" width="248" height="132" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="208" y="1374" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">lane 0 · 私有寄存器</text>
<rect x="92" y="1382" width="232" height="14" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="208" y="1392" font-size="9" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][1][0]</text>
<rect x="92" y="1398" width="232" height="14" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="208" y="1408" font-size="9" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][1][0]</text>
<rect x="92" y="1414" width="232" height="14" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="208" y="1424" font-size="9" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][1][0]</text>
<rect x="92" y="1430" width="232" height="14" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="208" y="1440" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][1][0]</text>
<rect x="92" y="1446" width="232" height="14" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="208" y="1456" font-size="9" fill="#b45309" font-weight="700" text-anchor="middle">P[j][1][0]</text>
<rect x="92" y="1462" width="232" height="14" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="208" y="1472" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][1][0]</text>
<rect x="340" y="1358" width="248" height="132" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="464" y="1374" font-size="9" fill="#334155" font-weight="700" text-anchor="middle">lane 1 · 私有寄存器</text>
<rect x="348" y="1382" width="232" height="14" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="464" y="1392" font-size="9" fill="#2563eb" font-weight="700" text-anchor="middle">Q[i][1][1]</text>
<rect x="348" y="1398" width="232" height="14" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="464" y="1408" font-size="9" fill="#16a34a" font-weight="700" text-anchor="middle">K[j][1][1]</text>
<rect x="348" y="1414" width="232" height="14" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="464" y="1424" font-size="9" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j][1][1]</text>
<rect x="348" y="1430" width="232" height="14" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="464" y="1440" font-size="9" fill="#dc2626" font-weight="700" text-anchor="middle">S[j][1][1]</text>
<rect x="348" y="1446" width="232" height="14" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="464" y="1456" font-size="9" fill="#b45309" font-weight="700" text-anchor="middle">P[j][1][1]</text>
<rect x="348" y="1462" width="232" height="14" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="464" y="1472" font-size="9" fill="#ea580c" font-weight="700" text-anchor="middle">O[i][1][1]</text>
<text x="72" y="1520" font-size="10" fill="#334155">Q 复用；S/P 暂存；O 与 m/l 随各轮更新。</text>
<path d="M608,1480 L632,1480 L632,912 L608,912" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="555" y="944" font-size="11" fill="#ea580c">O 写回</text>
<line x1="336" y1="1532" x2="336" y2="1546" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<line x1="344" y1="1548" x2="344" y2="1534" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<rect x="60" y="1548" width="560" height="52" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
<text x="340" y="1569" font-size="12" fill="#1d4ed8" font-weight="700" text-anchor="middle">MFMA：从各 lane 寄存器取 Q/K 或 P/V 元素</text>
<text x="340" y="1589" font-size="11" fill="#334155" text-anchor="middle">S / O 的计算结果回到各 lane 的寄存器</text>
<rect x="660" y="957" width="560" height="114" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="672" y="978" font-size="12" fill="#0f172a" font-weight="700">共享 SMEM：Q 常驻，K/V 随迭代载入</text>
<rect x="672" y="990" width="166" height="61" rx="3" fill="#dbeafe" stroke="#2563eb"/>
<text x="755.0" y="1007" font-size="11" fill="#2563eb" font-weight="700" text-anchor="middle">Q0 / Q1</text>
<text x="755.0" y="1024" font-size="10" fill="#334155" text-anchor="middle">各 128 × D</text>
<rect x="846" y="990" width="166" height="61" rx="3" fill="#dcfce7" stroke="#16a34a"/>
<text x="929.0" y="1007" font-size="11" fill="#16a34a" font-weight="700" text-anchor="middle">K[j]</text>
<text x="929.0" y="1024" font-size="10" fill="#334155" text-anchor="middle">128 × D</text>
<rect x="1020" y="990" width="188" height="61" rx="3" fill="#ede9fe" stroke="#7c3aed"/>
<text x="1114.0" y="1007" font-size="11" fill="#7c3aed" font-weight="700" text-anchor="middle">V[j]</text>
<text x="1114.0" y="1024" font-size="10" fill="#334155" text-anchor="middle">128 × D</text>
<line x1="940" y1="924" x2="940" y2="952" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="950" y="944" font-size="11" fill="#64748b">Q/K/V：TMA</text>
<line x1="940" y1="1071" x2="940" y2="1102" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="950" y="1095" font-size="11" fill="#64748b">tcgen05 → S / O</text>
<rect x="660" y="1106" width="560" height="92" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="672" y="1128" font-size="12" fill="#0f172a" font-weight="700">TMEM：S/P 与 O 的 block tile</text>
<rect x="672" y="1140" width="264" height="46" rx="3" fill="#fee2e2" stroke="#dc2626"/>
<text x="804.0" y="1157" font-size="11" fill="#dc2626" font-weight="700" text-anchor="middle">S/P · 对应 Q0 与 Q1</text>
<text x="804.0" y="1174" font-size="10" fill="#334155" text-anchor="middle">各 128 × 128</text>
<rect x="944" y="1140" width="264" height="46" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="1076.0" y="1157" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O(Q0) · O(Q1)</text>
<text x="1076.0" y="1174" font-size="10" fill="#334155" text-anchor="middle">各 128 × D</text>
<line x1="932" y1="1198" x2="932" y2="1236" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<line x1="940" y1="1240" x2="940" y2="1202" stroke="#64748b" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="952" y="1223" font-size="11" fill="#334155">按角色读写工作片段</text>
<rect x="660" y="1240" width="560" height="292" rx="6" fill="#ffffff" stroke="#cbd5e1"/>
<text x="672" y="1260" font-size="12" fill="#0f172a" font-weight="700">寄存器：每个 lane 私有</text>
<text x="672" y="1280" font-size="11" fill="#334155">一个角色 warp 有 32 个 lane；按角色持有所需工作元素。</text>
<rect x="672" y="1294" width="264" height="190" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="804" y="1312" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 0 · 私有寄存器</text>
<rect x="680" y="1324" width="248" height="34" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="804.0" y="1345.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">S/P 工作元素</text>
<rect x="680" y="1374" width="248" height="34" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="804.0" y="1395.0" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O 工作元素</text>
<rect x="680" y="1424" width="248" height="34" rx="3" fill="#f1f5f9" stroke="#64748b"/>
<text x="804" y="1446" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">m/l · 修正量</text>
<rect x="944" y="1294" width="264" height="190" rx="4" fill="#f8fafc" stroke="#cbd5e1"/>
<text x="1076" y="1312" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">lane 1 · 私有寄存器</text>
<rect x="952" y="1324" width="248" height="34" rx="3" fill="#fef3c7" stroke="#b45309"/>
<text x="1076.0" y="1345.0" font-size="11" fill="#b45309" font-weight="700" text-anchor="middle">S/P 工作元素</text>
<rect x="952" y="1374" width="248" height="34" rx="3" fill="#ffedd5" stroke="#ea580c"/>
<text x="1076.0" y="1395.0" font-size="11" fill="#ea580c" font-weight="700" text-anchor="middle">O 工作元素</text>
<rect x="952" y="1424" width="248" height="34" rx="3" fill="#f1f5f9" stroke="#64748b"/>
<text x="1076" y="1446" font-size="11" fill="#334155" font-weight="700" text-anchor="middle">m/l · 修正量</text>
<text x="672" y="1514" font-size="10" fill="#334155">工作元素随 softmax / correction / epilogue 等角色变化。</text>
<path d="M1208,1163 L1228,1163 L1228,912 L1208,912" fill="none" stroke="#ea580c" stroke-width="1.4" marker-end="url(#arr)"/>
<text x="1150" y="944" font-size="11" fill="#ea580c">O 写回</text>
<rect x="660" y="1548" width="560" height="52" rx="6" fill="#eff6ff" stroke="#93c5fd"/>
<text x="940" y="1569" font-size="11" fill="#1d4ed8" font-weight="700" text-anchor="middle">tcgen05：Q/K 来自 SMEM；P 来自 TMEM，V 来自 SMEM</text>
<text x="940" y="1589" font-size="11" fill="#334155" text-anchor="middle">S / O 的矩阵乘结果写入 TMEM</text>
<text x="60" y="1620" font-size="10" fill="#334155">寄存器元素分配为逻辑示意；实际布局由编译器决定。LDS / SMEM / TMEM 是共享存储，不按 lane 各复制一份。</text>
<g transform="translate(0,60)">
  <!-- ================= Summary table ================= -->
  <g transform="translate(30,1592)">
    <rect width="1220" height="140" fill="#ffffff" stroke="#cbd5e1" rx="10"/>
    <g class="txt" font-weight="700">
      <text x="20" y="26">参数</text><text x="110" y="26">落在哪层</text><text x="380" y="26">memory</text><text x="560" y="26">时间</text><text x="700" y="26">MFU</text>
    </g>
    <line x1="10" y1="36" x2="1210" y2="36" stroke="#e2e8f0"/>
    <rect class="chipB" x="20" y="44" width="34" height="18" rx="4"/><text x="37" y="57" class="chipT" text-anchor="middle">B</text>
    <text x="110" y="58" class="txt">① tile 个数</text><text x="380" y="58" class="txt">∝ B</text><text x="560" y="58" class="txt">∝ B</text><text x="700" y="58" class="txt">只影响能否喂饱 GPU；吃满后无关</text>
    <rect class="chipH" x="20" y="68" width="34" height="18" rx="4"/><text x="37" y="81" class="chipT" text-anchor="middle">H</text>
    <text x="110" y="82" class="txt">① tile 个数</text><text x="380" y="82" class="txt">∝ H</text><text x="560" y="82" class="txt">∝ H</text><text x="700" y="82" class="txt">同 B</text>
    <rect class="chipL" x="20" y="92" width="34" height="18" rx="4"/><text x="37" y="105" class="chipT" text-anchor="middle">L</text>
    <text x="110" y="106" class="txt">① tile 个数 + ② 循环长度</text><text x="380" y="106" class="txt">∝ L (不是 L²)</text><text x="560" y="106" class="txt">∝ L²</text><text x="700" y="106" class="txt">摊薄首尾开销；n = L/64 越大 prefetch 越赚</text>
    <rect class="chipD" x="20" y="116" width="34" height="18" rx="4"/><text x="37" y="129" class="chipT" text-anchor="middle">D</text>
    <text x="110" y="130" class="txt">④ 计算片段 + LDS 占用</text><text x="380" y="130" class="txt">∝ D</text><text x="560" y="130" class="txt">∝ D</text><text x="700" y="130" class="txt">D 小 → softmax 占比高 → MFU 低；D 大 → LDS 紧</text>
  </g>
  <rect class="lvl" x="30" y="1760" width="1220" height="488"/>
  <text x="48" y="1792" class="ttl">AMD 术语（NVIDIA 对应项）——功能对应不表示硬件完全相同</text>
  <text x="50" y="1826" class="txt" font-weight="700">workgroup（CTA / thread block）</text>
  <text x="50" y="1845" class="sm">线程块：一组协作线程</text>
  <text x="50" y="1877" class="txt" font-weight="700">wave / wavefront（warp）</text>
  <text x="50" y="1896" class="sm">线程执行组：此 AMD 为 64 线程，NV 为 32</text>
  <text x="50" y="1928" class="txt" font-weight="700">CU（SM，功能层级对应）</text>
  <text x="50" y="1947" class="sm">运行线程块的计算单元；不是完全相同的硬件</text>
  <text x="50" y="1979" class="txt" font-weight="700">XCD（无直接对应的编程层级）</text>
  <text x="50" y="1998" class="sm">AMD 计算 chiplet，内含多个 CU</text>
  <text x="50" y="2030" class="txt" font-weight="700">LDS（SMEM / shared memory）</text>
  <text x="50" y="2049" class="sm">一个线程块共享的片上内存；不是 TMEM</text>
  <text x="50" y="2081" class="txt" font-weight="700">寄存器（registers）</text>
  <text x="50" y="2100" class="sm">每线程私有的存储</text>
  <text x="50" y="2132" class="txt" font-weight="700">MFMA（MMA 矩阵乘指令；BW 用 tcgen05）</text>
  <text x="50" y="2151" class="sm">wave 发出指令，矩阵单元执行；不是 warp</text>
  <text x="50" y="2183" class="txt" font-weight="700">HBM（同名；global memory 的物理后端）</text>
  <text x="50" y="2202" class="sm">存放输入输出的显存</text>
  <text x="660" y="1826" class="txt" font-weight="700">async_load（功能类似异步拷贝 / TMA 搬运）</text>
  <text x="660" y="1845" class="sm">HBM → LDS；与 NVIDIA 的机制不同</text>
  <text x="660" y="1877" class="txt" font-weight="700">tl.load（普通全局内存 load）</text>
  <text x="660" y="1896" class="sm">此 kernel 读 Q 到寄存器，不走 TMA</text>
  <text x="660" y="1928" class="txt" font-weight="700">program_id（逻辑 block ID）</text>
  <text x="660" y="1947" class="sm">Triton 两边同名；常规 launch 类似 blockIdx</text>
  <text x="660" y="1979" class="txt" font-weight="700">slot / NUM_BUFFERS（buffer 槽 / 缓冲数）</text>
  <text x="660" y="1998" class="sm">交替保存当前和下一块 K/V</text>
  <text x="660" y="2030" class="txt" font-weight="700">num_warps（Triton 的执行组数量参数）</text>
  <text x="660" y="2049" class="sm">在 AMD 数 wave，在 NVIDIA 数 warp</text>
  <text x="660" y="2081" class="txt" font-weight="700">local_trans（局部布局转换，无直接同名项）</text>
  <text x="660" y="2100" class="sm">TLX 的布局转换功能；不是硬件执行组</text>
  <text x="660" y="2132" class="txt" font-weight="700">prefetch / prologue / epilogue（两边同名）</text>
  <text x="660" y="2151" class="sm">预取 / 流水启动 / 流水收尾</text>
  <text x="660" y="2183" class="txt" font-weight="700">BM / BN、Q/K/V、softmax（两边同义）</text>
  <text x="660" y="2202" class="sm">tile 高宽、输入矩阵、计算步骤；不是线程组</text>
</g>
</g>
</g>
</svg>
```


language: markdown

creation_time: 2026-10-15T22:51:47Z

update_time: 2026-10-15T22:51:47Z
