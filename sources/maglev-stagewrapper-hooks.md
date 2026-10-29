# maglev StageWrapper hook 连线（rank 2）与改写前 MaglevModuleList

这张图说明 rank 2 上的 StageWrapper 怎样通过 hook 与 MaglevLayer 连接：顶部是改写之前的完整 MaglevModuleList，构造期（蓝色）读取 spec 并在构造时切分，运行期（紫色）按 1–8 编号执行。

新 session 只需读本 paste。

- 基于：无

## 知识

| ID | 内容 | 来源 | 状态 |
|---|---|---|---|
| K1 | `StageWrapper.__init__` 用 `layers[0].in_activation_specs()` 生成 `_in_specs`（P2P 接收格式） | fbcode/torchrec/distributed/maglev/stage.py:956 | verified |
| K2 | `StageWrapper.__init__` 用 `layers[-1].out_activation_specs()` 生成 `_out_specs`，只在构造时读取一次 | fbcode/torchrec/distributed/maglev/stage.py:957 | verified |
| K3 | activation specs 在构造函数里、任何 wrap 之前捕获 | fbcode/torchrec/distributed/maglev/stage.py:775（docstring） | verified |
| K4 | `postproc` 只挂在末 stage 的 `_LayerChain` 上（`is_last_stage`） | fbcode/torchrec/distributed/maglev/stage.py:960, 965 | verified |
| K5 | `start_recv_grad()` 在末 stage 直接 return；否则按 `_out_specs` 中 `requires_grad` 的输出分配梯度 buffer，并从 `neighbor_rank(1)` irecv | fbcode/torchrec/distributed/maglev/stage.py:1277 | verified |
| K6 | `wait_for_grad()` 出队 `start_recv_grad` 投递的接收 | fbcode/torchrec/distributed/maglev/stage.py:1299 | verified |
| K7 | `_backward_activations` 用 `_out_specs` 的 `requires_grad` 过滤输出 | fbcode/torchrec/distributed/maglev/stage.py:1583 | verified |
| K8 | `_backward_round` 在 `wait_for_grad` 之后用 `_out_specs` 过滤 grad-carrying 输出 | fbcode/torchrec/distributed/maglev/stage.py:1736 | verified |
| K9 | `StageWrapper.module` 是可重新赋值的公开属性，调用方可以 wrap（DMP/FSDP/DDP），也可以不 wrap | fbcode/torchrec/distributed/maglev/stage.py:961, 755–758 | verified |
| K10 | 每个 rank 传入整个 model；StageWrapper 保留连续的一段层、丢弃其余；保留的是 model 自己的 module，不是拷贝 | fbcode/torchrec/distributed/maglev/stage.py:745–749（docstring） | verified |
| K11 | `_LayerChain` 存在的原因：`DistributedModelParallel` 只能包一个 module | fbcode/torchrec/distributed/maglev/stage.py:331–335（docstring） | verified |
| K12 | APS 侧 `build_maglev_stage`：每个 rank 传入整个 model，wrapper 丢掉本 rank 不拥有的层 | fbcode/aps_models/ads/common/maglev_pp/maglev_pp_runtime.py:240 | verified |
| K13 | APS 调用顺序：先 `rewrite_into_stage_bundles(model, this_stage=maglev_pp_stage_index(rank, ...))`，再 `build_maglev_stage(model=module_list, layers_per_stage=[1]*num_stages)` | fbcode/aps_models/ads/common/aps_maglev_pp_wrapper.py:605–636 | verified |
| K14 | `rewrite_into_stage_bundles` 内部是否只保留本 stage 的 bundle | 未读实现 | unverified |
| K15 | APS 上整模型是否在 meta 上构造 | aps_maglev_pp_wrapper.py 中未找到 meta 相关代码 | unverified |
| K16 | benchmark 中每个 rank 先在 meta 上 init 整模型，再建 StageWrapper | fbcode/torchrec/distributed/maglev/benchmark/benchmark_maglev.py ~370–446（session 1bacaabf-1c6d-54d4-a008-aedc25446074 @ 2026-10-29） | verified |
| K17 | 运行期 stage_input 来自 cascade 上的 all-to-all（集合通信），落在本地 buffer；wait_for_act 是从 rank 0 的 P2P irecv | input_dist.py ~595；fbcode/torchrec/distributed/maglev/stage.py:1190–1222（session 1bacaabf-1c6d-54d4-a008-aedc25446074 @ 2026-10-29） | verified |
| K18 | rank 2 是 stage 1（末 stage），持有 L2、L3；stage 0 为 rank 0、1，持有 L0、L1 | aps_maglev_pp_wrapper.py:609（`maglev_stages.{i}`）；前序对话绘制的 before-after 图 | inferred |
| K19 | `build_maglev_stage` 在 stage 有多个 rank 时用 DistributedDataParallel 包 stage.module（stage 内数据并行） | fbcode/aps_models/ads/common/maglev_pp/maglev_pp_runtime.py:280–300 | 未画入：本图只表示 rank 2 的 hook 连线，不画 stage 内 DDP 副本关系 |

状态：verified = 已对照源码或文档；inferred = 由代码或数字推出；unverified = 只来自对话，未核对；未画入 = 有依据但图里没画，在备注里写原因。

## 结构

- 实体：rank 0 与 cascade 同 position 的 rank（外框）；StageWrapper · rank 2；改写之前的 MaglevModuleList（L0–L3）；`_LayerChain` 中的 MaglevLayer L2、L3（各有 in_activation_specs、out_activation_specs、forward 三个框）；wait_for_act、wait_for_grad、stage_input[0]/[1]、postproc、loss.backward、_pending、optimizer.step。对应 K1–K8、K10、K17
- 关系：蓝色虚线 `StageWrapper.__init__` → L2 in_activation_specs；蓝色虚线 `读 L3.out_spec` → L3 out_activation_specs；蓝色虚线 out_activation_specs → wait_for_grad（生成 P2P grad expect 格式）；蓝色虚线 L2 in_activation_specs → wait_for_act（生成 P2P expect 输入格式）；顶部蓝色虚线 L2、L3 → StageWrapper（切分，K9、K10）；紫色实线 forward 串接、P2P 收发、loss.backward 回传、optimizer.step 收尾。对应 K1、K2、K5–K8、K17
- 分组与颜色：蓝色 = 构造期（只执行一次）；紫色 = 运行期（每个 microbatch，按编号 1–8）；灰色虚线框 = 丢弃的层，或 rank 2 为末 stage 时跳过的 wait_for_grad；顶部区域 = 改写之前的完整 MaglevModuleList；StageWrapper 外框 = rank 2 持有的 stage。
- 阅读顺序：先看顶部改写前的 list（L0–L3），再看 StageWrapper 中的构造期（蓝色，从左上到右上），最后按 1 → 8 看运行期。

## SVG 约定

- 尺寸：`viewBox="0 0 1580 1140"`，渲染宽度 1240
- id：本版 SVG 没有 node-/edge-/group- id 和 data-k（v3 早于本约定），见未解决
- 渲染方法：见 img-context-paste skill 的构建步骤第 5 步（resvg-js，已用于本版渲染检查）

## 修改记录

- 2026-10-29 画出 v3：补上 out_activation_specs 构造期到 wait_for_grad 的连线；在顶部加入改写前的完整 MaglevModuleList；撤销误加的"不 wrap"框和 DMP 框（基于 无，session 1bacaabf-1c6d-54d4-a008-aedc25446074）

## 未解决

- [ ] K14：确认 `rewrite_into_stage_bundles` 是否只保留本 stage 的 bundle
- [ ] K15：确认 APS 上整模型是否在 meta 上构造；确认前，图中不写 meta
- [ ] APS 侧的 bundle rewrite（K13）没有画进顶部区域
- [ ] SVG 缺 node-/edge-/data-k 标注，下次修改前先补
- [ ] `/home/irobert/maglev_diagrams/` 下已存的 .svg/.png 是否删除（本 skill 要求只保留 paste），需要你确认

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1580 1140" width="1580" height="1140" role="img">
<title>改写之后：StageWrapper 与 MaglevLayer 的 hook 连线（rank 2）：构造期蓝色，运行期紫色，区分 P2P 与集合通信</title>
<desc>MaglevLayer 内的 in_activation_specs、out_activation_specs、forward 各自为一个框。构造期 StageWrapper.__init__ 调用 L2 的 in_activation_specs，生成 P2P 接收格式，再交给 wait_for_act。运行期 stage_input 来自 cascade 上的 all-to-all，结果落在本地 buffer；wait_for_act 是从 rank 0 的 P2P irecv；forward、_pending、loss.backward、in_acts.grad 经 P2P isend 发回 rank 0，最后 optimizer.step。</desc>
<defs>
<marker id="arrP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" style="fill:#a855f7"/></marker>
<marker id="arrB" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" style="fill:#4f7cff"/></marker>
<marker id="arrG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10z" style="fill:#6b7280"/></marker>
</defs>

<!-- Title and legend -->
<text x="20" y="36" style="font:600 22px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">改写之后：StageWrapper 与 MaglevLayer 的 hook 连线（rank 2）</text>
<text x="20" y="62" style="font:14px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">蓝色 = 构造期（只执行一次，虚线）；紫色 = 运行期（每个 microbatch，按编号 1–8，实线）</text>
<rect x="20" y="78" width="18" height="18" rx="3" style="fill:#4f7cff;fill-opacity:.25;stroke:#4f7cff"/>
<text x="46" y="92" style="font:14px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">构造期</text>
<rect x="130" y="78" width="18" height="18" rx="3" style="fill:#a855f7;fill-opacity:.25;stroke:#a855f7"/>
<text x="156" y="92" style="font:14px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">运行期</text>
<circle cx="250" cy="87" r="6" style="fill:#64748b"/>
<text x="264" y="92" style="font:14px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">hook 端口</text>

<!-- Before rewrite: the whole MaglevModuleList on rank 2, handed to StageWrapper and cut there -->
<rect x="20" y="100" width="1540" height="180" rx="10" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="34" y="126" style="font:600 15px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">改写之前：rank 2 上的完整 MaglevModuleList</text>
<text x="34" y="146" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">构造期（蓝色）：整个 list 传入 StageWrapper，在构造时切分</text>
<rect x="40" y="160" width="360" height="100" rx="6" style="fill:none;stroke:#6b7280;stroke-dasharray:4 3"/>
<text x="54" y="186" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#6b7280">MaglevLayer · L0</text>
<text x="54" y="210" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">stage 0 的层，rank 2 不持有</text>
<text x="54" y="232" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">切分时丢弃</text>
<rect x="420" y="160" width="360" height="100" rx="6" style="fill:none;stroke:#6b7280;stroke-dasharray:4 3"/>
<text x="434" y="186" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#6b7280">MaglevLayer · L1</text>
<text x="434" y="210" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">stage 0 的层，rank 2 不持有</text>
<text x="434" y="232" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">切分时丢弃</text>
<rect x="800" y="160" width="360" height="100" rx="6" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="814" y="186" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">MaglevLayer · L2</text>
<text x="814" y="210" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">stage 1 的层，rank 2 持有</text>
<text x="814" y="232" style="font:600 12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">切分后进入 _LayerChain</text>
<rect x="1180" y="160" width="360" height="100" rx="6" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="1194" y="186" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">MaglevLayer · L3</text>
<text x="1194" y="210" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">stage 1 的层，rank 2 持有</text>
<text x="1194" y="232" style="font:600 12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">切分后进入 _LayerChain</text>
<path d="M900 260V306" style="stroke:#4f7cff;stroke-width:1.8;stroke-dasharray:5 4;fill:none" marker-end="url(#arrB)"/>
<text x="914" y="300" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">切分：L2、L3 进入 StageWrapper</text>

<!-- Everything below the legend shifts down to make room for the band above -->
<g transform="translate(0,200)">
<!-- StageWrapper rank 2, shifted right to make room for the other-rank boxes -->
<g transform="translate(240,0)">
<!-- Outer: StageWrapper rank 2 -->
<rect x="20" y="110" width="1300" height="800" rx="10" style="fill:#ffffff;stroke:#d0d7de"/>
<path d="M20 140V118Q20 110 28 110H1312Q1320 110 1320 118V140Z" style="fill:#6b7280;fill-opacity:.18"/>
<text x="34" y="132" style="font:600 16px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">StageWrapper · rank 2（stage 1, position 0）</text>

<!-- Construction phase (blue) -->
<text x="40" y="176" style="font:600 15px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">构造期（只执行一次）</text>
<rect x="40" y="186" width="420" height="80" rx="6" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="54" y="212" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">StageWrapper.__init__()</text>
<text x="54" y="234" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">读 L2.in_spec；check L2.out == L3.in</text>
<text x="54" y="252" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">→ _in_specs（P2P 接收格式）</text>
<rect x="560" y="186" width="420" height="80" rx="6" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="574" y="212" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">读 L3.out_spec</text>
<text x="574" y="234" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">→ _out_specs（P2P 梯度格式）</text>
<text x="574" y="252" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">只在 init 时读取一次</text>

<!-- Construction wires: blue dashed, into the in/out spec boxes inside each layer -->
<path d="M330 266V370" style="stroke:#4f7cff;stroke-width:1.5;stroke-dasharray:5 4;fill:none" marker-end="url(#arrB)"/>
<path d="M955 266V430H938" style="stroke:#4f7cff;stroke-width:1.5;stroke-dasharray:5 4;fill:none" marker-end="url(#arrB)"/>

<!-- _LayerChain container -->
<rect x="280" y="290" width="1020" height="350" rx="10" style="fill:none;stroke:#6b7280"/>
<text x="1286" y="314" text-anchor="end" style="font:600 15px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">_LayerChain（module，DMP 包住它）</text>

<!-- L2: MaglevLayer instance with one box per API -->
<rect x="300" y="330" width="300" height="260" rx="8" style="fill:#ffffff;stroke:#d0d7de"/>
<path d="M300 362V338Q300 330 308 330H592Q600 330 600 338V362Z" style="fill:#6b7280;fill-opacity:.18"/>
<text x="450" y="352" text-anchor="middle" style="font:600 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">MaglevLayer · L2</text>
<rect x="312" y="370" width="276" height="40" rx="5" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="324" y="388" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">in_activation_specs()</text>
<text x="324" y="404" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">输入 spec ← L1 的输出</text>
<rect x="312" y="420" width="276" height="40" rx="5" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="324" y="438" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">out_activation_specs()</text>
<text x="324" y="454" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">输出 spec → L3 in</text>
<rect x="312" y="470" width="276" height="62" rx="5" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="324" y="488" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">forward(layer_input, in_acts)</text>
<text x="324" y="506" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">layer_input ← stage_input[0]</text>
<text x="324" y="522" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">in_acts ← rank 0；→ Activations</text>

<!-- L3: MaglevLayer instance with one box per API -->
<rect x="650" y="330" width="300" height="260" rx="8" style="fill:#ffffff;stroke:#d0d7de"/>
<path d="M650 362V338Q650 330 658 330H942Q950 330 950 338V362Z" style="fill:#6b7280;fill-opacity:.18"/>
<text x="800" y="352" text-anchor="middle" style="font:600 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">MaglevLayer · L3</text>
<rect x="662" y="370" width="276" height="40" rx="5" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="674" y="388" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">in_activation_specs()</text>
<text x="674" y="404" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">输入 spec ← L2 的输出</text>
<rect x="662" y="420" width="276" height="40" rx="5" style="fill:#4f7cff;fill-opacity:.10;stroke:#4f7cff"/>
<text x="674" y="438" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">out_activation_specs()</text>
<text x="674" y="454" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">输出 spec → _out_specs（末层）</text>
<rect x="662" y="470" width="276" height="62" rx="5" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="674" y="488" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">forward(layer_input, in_acts)</text>
<text x="674" y="506" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">layer_input ← stage_input[1]</text>
<text x="674" y="522" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">in_acts ← L2；→ 交给 postproc</text>

<!-- Blue: out_activation_specs() output feeds the P2P grad receive (non-last stages only) -->
<path d="M938 445H1000" style="stroke:#4f7cff;stroke-width:1.8;fill:none" marker-end="url(#arrB)"/>
<text x="1000" y="394" style="font:11px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">生成了P2P grad expect的格式</text>
<rect x="1000" y="400" width="280" height="64" rx="6" style="fill:none;stroke:#6b7280;stroke-dasharray:5 4"/>
<text x="1012" y="420" style="font:600 12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#6b7280">wait_for_grad()</text>
<text x="1012" y="438" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">非末 stage：P2P irecv 梯度</text>
<text x="1012" y="454" style="font:11.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">rank 2 为末 stage：跳过</text>

<!-- Runtime hook ports (purple) on the forward boxes -->
<circle cx="300" cy="500" r="6" style="fill:#a855f7"/>
<circle cx="600" cy="500" r="6" style="fill:#a855f7"/>
<circle cx="650" cy="500" r="6" style="fill:#a855f7"/>
<circle cx="950" cy="500" r="6" style="fill:#a855f7"/>
<circle cx="360" cy="532" r="6" style="fill:#a855f7"/>
<circle cx="800" cy="532" r="6" style="fill:#a855f7"/>

<!-- Runtime wires (purple, solid) -->
<path d="M250 500H300" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<path d="M300 520H250" style="stroke:#a855f7;stroke-width:1.8;stroke-dasharray:5 4;fill:none" marker-end="url(#arrP)"/>
<path d="M600 500H650" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<path d="M950 500H1000" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<path d="M360 690V534" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<path d="M800 690V534" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<path d="M1000 614H936V593" style="stroke:#a855f7;stroke-width:1.8;stroke-dasharray:5 4;fill:none" marker-end="url(#arrP)"/>

<!-- Blue: in_activation_specs() output feeds wait_for_act (P2P expected format) -->
<path d="M312 390H250" style="stroke:#4f7cff;stroke-width:1.8;fill:none" marker-end="url(#arrB)"/>
<text x="46" y="368" style="font:11px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#4f7cff">生成了P2P expect的input格式</text>

<!-- Recv: wait_for_act = P2P irecv from rank 0 -->
<rect x="40" y="380" width="210" height="150" rx="6" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="54" y="404" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">wait_for_act()</text>
<text x="54" y="426" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">P2P 收到 in_acts</text>
<text x="54" y="446" style="font:600 12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">→ 开始 train</text>

<!-- Sources: stage_input = cascade all-to-all result in local buffer -->
<rect x="250" y="690" width="220" height="56" rx="6" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="262" y="708" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">stage_input[0]</text>
<text x="262" y="725" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">cascade all-to-all 结果</text>
<text x="262" y="741" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">（本地 buffer）</text>
<rect x="690" y="690" width="220" height="56" rx="6" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="702" y="708" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">stage_input[1]</text>
<text x="702" y="725" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">cascade all-to-all 结果</text>
<text x="702" y="741" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">（本地 buffer）</text>

<!-- Postproc (last stage) -->
<rect x="1000" y="470" width="280" height="80" rx="6" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="1012" y="494" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">postproc(acts, layer_input[-1])</text>
<text x="1012" y="518" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">末 stage：返回 (losses, output)</text>

<!-- Backward and pending -->
<rect x="1000" y="594" width="280" height="40" rx="6" style="fill:#ffffff;stroke:#a855f7"/>
<text x="1012" y="619" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">loss.backward()</text>
<rect x="940" y="690" width="340" height="56" rx="6" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="952" y="712" style="font:13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">_pending.append((in_acts, out, id))</text>
<text x="952" y="732" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">backward 按 FIFO 取回同一张图</text>

<!-- Optimizer -->
<rect x="480" y="690" width="200" height="56" rx="6" style="fill:#ffffff;stroke:#d0d7de"/>
<text x="492" y="712" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">optimizer.step()</text>
<text x="492" y="732" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">pass 末，一次</text>

<!-- Step badges (runtime order) -->
<circle cx="275" cy="478" r="12" style="fill:#a855f7"/>
<text x="275" y="483" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">2</text>
<circle cx="275" cy="540" r="12" style="fill:#a855f7"/>
<text x="275" y="545" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">7</text>
<circle cx="625" cy="500" r="12" style="fill:#a855f7"/>
<text x="625" y="505" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">3</text>
<circle cx="975" cy="500" r="12" style="fill:#a855f7"/>
<text x="975" y="505" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">4</text>
<circle cx="388" cy="664" r="12" style="fill:#a855f7"/>
<text x="388" y="669" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">1</text>
<circle cx="828" cy="664" r="12" style="fill:#a855f7"/>
<text x="828" y="669" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">1</text>
<circle cx="1262" cy="708" r="12" style="fill:#a855f7"/>
<text x="1262" y="713" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">5</text>
<circle cx="1262" cy="614" r="12" style="fill:#a855f7"/>
<text x="1262" y="619" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">6</text>
<circle cx="662" cy="712" r="12" style="fill:#a855f7"/>
<text x="662" y="717" text-anchor="middle" style="font:700 13px system-ui,-apple-system,sans-serif;fill:#ffffff">8</text>

<!-- Ordered explanation -->
<text x="40" y="800" style="font:600 15px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">运行期顺序（每个 microbatch）</text>
<text x="40" y="824" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">1</text>
<text x="62" y="824" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">stage_input：cascade all-to-all（集合通信）结果，落在本地 buffer</text>
<text x="40" y="846" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">2</text>
<text x="62" y="846" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">P2P irecv 收到 rank 0 的 in_acts → 开始 train（forward）</text>
<text x="40" y="868" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">3</text>
<text x="62" y="868" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">L2.forward 输出 Activations，串接给 L3.forward</text>
<text x="40" y="890" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">4</text>
<text x="62" y="890" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">L3 输出交给 postproc，得到 (losses, output)</text>
<text x="700" y="824" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">5</text>
<text x="722" y="824" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">_pending.append 保存计算图，供 backward 取回</text>
<text x="700" y="846" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">6</text>
<text x="722" y="846" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">loss.backward()：沿 L3 → L2 反传</text>
<text x="700" y="868" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">7</text>
<text x="722" y="868" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">in_acts.grad 经 start_send_grad（P2P isend）发回 rank 0</text>
<text x="700" y="890" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#a855f7">8</text>
<text x="722" y="890" style="font:12.5px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#111827">一个 pass 结束后 optimizer.step()</text>
</g>

<!-- Other ranks (outside StageWrapper): P2P peer and cascade peers -->
<rect x="20" y="380" width="200" height="64" rx="6" style="fill:#a855f7;fill-opacity:.10;stroke:#a855f7"/>
<text x="32" y="404" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">rank 0（stage 0）</text>
<text x="32" y="426" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">P2P isend → rank 2</text>
<path d="M220 412H280" style="stroke:#a855f7;stroke-width:2;fill:none" marker-end="url(#arrP)"/>
<text x="226" y="404" style="font:600 11px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#a855f7">P2P</text>

<rect x="20" y="690" width="200" height="56" rx="6" style="fill:#6b7280;fill-opacity:.10;stroke:#d0d7de"/>
<text x="32" y="712" style="font:600 13px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;fill:#111827">同 position 的 rank</text>
<text x="32" y="732" style="font:12px system-ui,-apple-system,'PingFang SC','Noto Sans CJK SC',sans-serif;fill:#6b7280">cascade all-to-all</text>
<path d="M220 718H490" style="stroke:#6b7280;stroke-width:1.8;fill:none" marker-end="url(#arrG)"/>
</g>
</svg>
```
