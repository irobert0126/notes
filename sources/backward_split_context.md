# PP 里 backward 怎么拆成 dI / dW：eager 不拆、eager 拆、compile 没有 fix、compile 有 fix

嵌套 paste P2548053087（已提取）：[backward_split_measurements.md](backward_split_measurements.md)

这张图解释 pipeline parallel（zero-bubble 调度）里怎么把一个 microbatch 的 backward 拆成 dI / dW（I / W），以及为什么旧实现只能在 eager 下工作、SplitBackwardBackend 怎么修好它。

新 session 只需读本 paste。

- 基于：无
- 对应文件（用户保留）：`~/diagrams/backward_split_iw/backward_four_cases.svg`、`backward_four_cases.png`，内容与本 paste 末尾的 SVG 源码相同

## 核心论点

**一句话目的：** 这张图解释 PP 的 split dI / dW 是怎么做的。

**核心知识点（用户原意，已润色）：**

之前的实现只能在 eager 模式下工作。原因是：拆分 I / W 的结果，是一个 microbatch 的 backward 变成两个 GraphTask（#1 I、#2 W，顺序执行），而这两个 GraphTask 都需要中间梯度 dh。之前的实现借 engine 的 InputBuffer 拿到 dh：dh 进入 fc1 的 InputBuffer 时被 prehook 截下，引用存进 param_groups，W 再把它喂回一个新的 InputBuffer。

compile 模式下，按 op 划分的多个 autograd 节点（fc1、fc2 的 `MmBackward0` 和 `TBackward0`）被合成一个 `CompiledFunctionBackward` 节点，dh 不再是节点之间的中间变量，也就不会存进 InputBuffer。prehook 截不到 dh，W 只能触发重算。

解法是让拆出来的 dI 图把 dh 作为输出交出来，存进节点的 ctx，由 ctx 把 dh 传给 W。

**整张图的精髓：** compile 没有 fix 那张图里，GraphTask #1 的红框「dh 不在任何 InputBuffer 里」。重算是从它推出来的：

1. 拆 I / W → 两个顺序执行的 GraphTask，都要用 dh（K1、K2）
2. 旧实现靠 InputBuffer 截取 dh、靠 param_groups 留到 W（K3–K11）
3. 前提是 dh 走在两个 autograd 节点之间的边上；eager 下每个 op 一个节点，前提成立（K12）
4. compile 后整个 dense half 是一个节点，dh 成了编译图内部的 buffer，不进任何 InputBuffer（K14、K15）
5. prehook 只截到 dy → param_groups 只有 dy → W 手里没有 dh，只能整张重跑（K16）
6. 有 fix：dI 图把 dh 交出来存进 ctx，W 只跑 dW 图、数据从 ctx 拿（K17、K18、K20）

**措辞说明：** dI、dW 本来不是两个独立的 autograd 节点，而是每个 `MmBackward0` 的两个输出；compile 合并的是按 op 划分的节点。InputBuffer 只在 I 那次调用里存在，是截取 dh 的地方；真正跨两个 GraphTask 拿着 dh 的是 param_groups 里的引用。

## 术语

| 术语 | 含义 |
|---|---|
| I / W（dI / dW） | 把一个 microbatch 的 backward 拆成两半：I 只算对输入的梯度（dgrad，dx 要尽快发给上一个 stage），W 只算对权重的梯度（wgrad，可以往后挪去填 pipeline bubble） |
| dgrad / wgrad | 一个 GEMM 反向的两个输出：对输入的梯度 / 对权重的梯度。本例前向 2 个 GEMM，反向 4 个 GEMM |
| dh | fc2 对它的输入 h 的梯度，`dh = dy @ W2`。I 的 dgrad1 和 W 的 wgrad1 都要用它 |
| autograd 节点（Node） | forward 时建好的反向图里的节点，backward 时由 engine 逐个执行。eager 下每个 aten op 一个，如 `MmBackward0`、`TBackward0`、`AccumulateGrad` |
| 边（Edge） | `next_functions` 里的一项 = (下一个节点, input_nr)。只有沿边流动的梯度才会进下一个节点的 InputBuffer |
| GraphTask | engine 为每次 `autograd.grad` / `backward` 调用建的执行上下文，里面有 exec_info、dependencies_、not_ready_（各节点的 InputBuffer） |
| InputBuffer / slot | engine 给还没执行的节点准备的输入槽，slot 个数 = 节点的 num_inputs，slot 号 = 边上的 input_nr；同一个 slot 收到多个梯度会相加；节点执行时随 NodeTask 被取走、用完释放 |
| exec_info | GraphTask 里"这次要执行哪些节点、截取哪些边"。`backward()` 不带 inputs 时为空（全部执行）；`autograd.grad(inputs=...)` 时只执行标记的节点 |
| prehook | 用 `Node.register_prehook` 挂在节点上的回调，节点执行前拿到 InputBuffer 里的梯度 |
| param_groups / intermediate | `stage_backward_input` 返回的分组：每组记下权重和"权重路径与输入路径交汇的节点"（intermediate），以及 prehook 在这个节点上截到的梯度引用 |
| GradientEdge / 根梯度 | W 阶段 `autograd.grad` 的起点 = (intermediate 节点, 输入编号)，配上存下的梯度作为根梯度 |
| AccumulateGrad | 叶子 tensor（权重、从上一个 stage 收到的激活 x）的梯度汇入节点；`autograd.grad` 截取时不往 `.grad` 里累加 |
| CompiledFunctionBackward | AOTAutograd 把整个编译后的 backward 包成一个 `autograd.Function`，autograd 里就是这一个节点，每次调用整张编译图跑完 |
| Inductor buffer（bufN） | 编译图内部的中间值，生成的代码在同一次调用里分配、使用、`del` |
| SplitBackwardBackend | D124884069 加的 `torch.compile` backend：导出 joint graph → min-cut 切出 forward / backward → backward 再切成 dI 图和 dW 图，分别编译 |
| `_SplitBackwardFunctionBackward` | SplitBackwardBackend 生成的那一个 autograd 节点；被调用时先问 engine 每个输入要不要（`torch._C._will_engine_execute_node`），再决定只跑 dI 图还是 dW 图 |
| live-ins / `ctx.weight_live_ins` | dI 图交给 dW 图的值（dy、h、x、dh），detach 后存在这个节点的 ctx 上，活到 W |
| ViewBackward0（`_engine_visible`） | 实现里给叶子输入包一层 `view_as`；因为 engine 不肯回答"会不会把梯度交给一个被 `autograd.grad` 截取的叶子" |

## 背景

- **为什么要拆 I / W：** dx 在 pipeline 的关键路径上，要尽快发给上一个 stage；wgrad 没人等着用。拆开以后每一跳的代价从 I + W 变成 I，W 往后挪去填 bubble。Maglev Rail 的 `_dense_zero_bubble` 用 ZB1P 规则，W 落后 I 的步数等于 stage index（K23）。
- **Maglev 里怎么调：** 不拆的 Maglev1F1B 用 `stage.backward_micro` → `torch.autograd.backward`；Rail 用 `dense_backward_act_micro`（I，`stage_backward_input`）和 `dense_backward_weight_micro`（W，`stage_backward_weight`）（K13）。dense half 被 FMC / `torch.compile` 以后就落到"没有 fix"那张图的情况。
- **例子：** 两层没有 bias 的 Linear（fc1 → fc2），前向 2 个 GEMM，反向 4 个 GEMM。真实模型的 dense half 更大，但机制一样。
- **实测（GB200，batch 16384 = 2 × 8192，dim 4096，fp32，P2548053087）：** I+W 延迟 eager 37.66 ms，没有 fix 73.60 ms，有 fix 37.00 ms（K16、K20）。
- **显存：** eager 拆分和有 fix 都要把一份 dh 从 I 留到 W，只是拿着它的地方不同（param_groups 的引用 vs ctx）；没有 fix 时不留 dh，代价是 W 重算。
- **不编译也能用 ctx：** `SplitBackwardBackend(eager_compiler)` 只做 trace、不交给 Inductor，按 aten op 逐个执行拆好的图，GPU 上 37.64 ms，和 eager 一样（K21）。纯 eager 不需要 ctx，因为节点足够细，dh 天然在边上。
- **LLM 侧的同类做法：** torchtitan GraphPP 的 `split_di_dw_graph` 也是在 FX 图上切出 bw_di / bw_dw，接口和 SplitBackwardBackend 基本一致（K22）。
- **相关 diff：** D117438870（Rail zero-bubble dense backward）、D124884069（SplitBackwardBackend）、D124884070（单测）、D125041326（msco maglev_pp 接入）。

## 知识

| ID | 内容 | 来源 | 状态 |
|---|---|---|---|
| K1 | 拆 I / W 后，一个 microbatch 的 backward 是两次 autograd 调用 = 两个顺序执行的 GraphTask（#1 I、#2 W），中间隔着其他 microbatch 的 F / I | `fbcode/torchrec/distributed/maglev/stage.py`（`dense_backward_act_micro` / `dense_backward_weight_micro`）、`pipeline.py`（`MaglevRail._dense_zero_bubble`）；session 1e0800bb @ 2026-10-12 | verified |
| K2 | I 的 dgrad1 和 W 的 wgrad1 都要用 dh，所以两个 GraphTask 都需要 dh | 反向公式 `dx = dh @ W1`、`dW1 = dhᵀ @ x` | inferred |
| K3 | 节点执行完，engine 对每个输出找 `next_edge(i) = (节点, input_nr)`，`add` 进下一个节点 InputBuffer 的对应 slot；`dependencies_` 归零就把 (节点, InputBuffer) 打包成 NodeTask 执行 | `fbcode/caffe2/torch/csrc/autograd/engine.cpp:1277-1344` | verified |
| K4 | InputBuffer 挂在 GraphTask 上，节点执行时随 NodeTask 被取走、用完释放；同一 slot 多个梯度相加，空梯度忽略 | `engine.cpp:1338-1341`、`input_buffer.cpp:350` 起 | verified |
| K5 | 节点执行前，engine 先对 InputBuffer 里的梯度调用 `call_pre_hooks`，再执行节点 | `engine.cpp:899`、`engine.cpp:1125-1128` | verified |
| K6 | 只有一条根边时，engine 不建 GraphRoot，直接给根节点新建一个 InputBuffer 放入传入的梯度 | `engine.cpp:1483-1520` | verified |
| K7 | `backward()` 不带 inputs 时 exec_info 为空 → 所有节点执行、每个节点所有输出都算；`autograd.grad(inputs=...)` 只执行标记的节点 | `engine.cpp:1302-1308` | verified |
| K8 | `stage_backward_input` 在每个 param group 的 intermediate 节点挂 prehook，把流入的梯度引用存进 `param_group["grads"]`，再 `autograd.grad(..., retain_graph=True)` | `fbcode/caffe2/torch/distributed/pipelining/_backward.py:182-301` | verified |
| K9 | `stage_backward_weight` 对每个 param group 调一次 `autograd.grad`，根是 `GradientEdge(intermediate, i)`，梯度是存下的 grads，用完 `del` | `_backward.py:304-366` | verified |
| K10 | `get_param_groups` 按权重路径与输入路径交汇的节点分组；本例 eager 分两组（W2 截在 fc2 存 dy，W1 截在 fc1 存 dh），compile 后只剩一组（截在整个编译节点，存 dy） | `_backward.py:93-144`；分组结果按本例推出 | inferred |
| K11 | prehook 存的是同一个 tensor 的引用，不是拷贝；dh 只有一份，先后出现在 #1、#2 的 InputBuffer 里 | 由 K5、K8、K9 推出 | inferred |
| K12 | eager 下每个 aten op 一个 autograd 节点；无 bias 的 Linear 反向 = `MmBackward0` + 权重转置的 `TBackward0` | ATen `linear` 组合实现（本 session 未读源码） | inferred |
| K13 | 不拆的 Maglev1F1B 用 `stage.backward_micro` → `torch.autograd.backward`，整次 backward 返回后才把 `x.grad` 发给上一个 stage | `fbcode/torchrec/distributed/maglev/stage.py`（`backward_micro`）；session 1e0800bb @ 2026-10-12 | verified |
| K14 | compile 后 AOTAutograd 把整个 dense half 的反向包成一个 `autograd.Function`，autograd 里只有一个 `CompiledFunctionBackward` 节点，每次调用整张编译图跑完 | `split_backward.py` 文档注释；D124884070 单测 matmul 计数；P2548053087 trace | verified |
| K15 | 编译图里的中间值（含 dh）是 Inductor buffer，生成的 wrapper 在同一次调用里 `del` | `fbcode/caffe2/torch/_inductor/codegen/wrapper.py:5925`（`make_buffer_free`）；bufN 是示意名 | verified |
| K16 | 没有 fix：I、W 各 4 个 GEMM，共 8 个只有 4 个有用；I+W 73.60 ms（eager 37.66 ms） | D124884070 单测；P2548053087 | verified |
| K17 | SplitBackwardBackend 把 backward 切成 dI 图、dW 图分别编译，挂在一个 autograd 节点后；节点用 `torch._C._will_engine_execute_node` 问每个输入要不要，I 只跑 dI 图，W 只跑 dW 图 | `fbcode/torchrec/distributed/maglev/split_backward.py`（D124884069） | verified |
| K18 | dI 图把 dh 等 live-in 作为输出交出，detach 后存进 `ctx.weight_live_ins`；W 只跑 dW 图，数据从 ctx 拿；param_groups 里的 dy 只用来把节点叫起来 | `split_backward.py`（`_SplitBackwardFunction.backward`） | verified |
| K19 | 叶子输入先包一层 `view_as`（`_engine_visible`），节点后面接 `ViewBackward0`；因为 engine 对被 `autograd.grad` 截取的叶子不回答 | `split_backward.py`；`fbcode/caffe2/torch/csrc/Module.cpp:1825-1828` | verified |
| K20 | 有 fix：I、W 共 4 个 GEMM；I+W 37.00 ms，和 eager 一样 | D124884070 单测；P2548053087 | verified |
| K21 | `SplitBackwardBackend(eager_compiler)`（只 trace、不编译、用 ctx 交接）也能跑通，37.64 ms | D124884070 单测；P2548053087（after_aten） | 未画入 |
| K22 | torchtitan GraphPP 的 `split_di_dw_graph` 是同一思路的 LLM 实现 | `fbcode/pytorch/torchtitan/torchtitan/experiments/graph_trainer/graph_pp/split_di_dw.py` | 未画入 |
| K23 | 拆 I / W 的动机：dx 在关键路径上，W 往后挪填 bubble；Maglev 用 ZB1P，W 落后步数 = stage index | `pipeline.py`（`MaglevRail` 文档注释）；session 1e0800bb @ 2026-10-12 | 未画入 |

状态：verified = 已对照源码或文档；inferred = 由代码或数字推出；unverified = 只来自对话，未核对；未画入 = 有依据但图里没画（K21–K23 是背景，写在「背景」里）。

## 图的内容

四张小图从上到下，结构相同：上层 autograd 图（forward 时建好），中层 autograd engine（GraphTask 和 InputBuffer 的 slot），下层 param_groups（Python 侧对象）和说明框。

1. **eager、没有 split**（`group-panel-eager-nosplit`）：一次 `torch.autograd.backward`，只有 GraphTask #1。dy → fc2 的 InputBuffer → fc2 执行，dgrad2、wgrad2 都算 → dh 存进 fc1 InputBuffer 的 slot 0 → fc1 执行一次，dgrad1、wgrad1 都从这个 slot 拿 dh → InputBuffer 释放，dh 随即释放。代价：dx 要等整次 backward（4 个 GEMM）跑完才发出。
2. **eager、split 成 I / W**（`group-panel-eager-engine`）：#1 只算 dgrad；fc1 执行前 prehook 把 dh 的引用存进 param_groups；#1 结束、fc1 的 InputBuffer 释放后，只剩 param_groups 拿着 dh；#2 把 dh 作为根梯度喂进新建的 fc1 InputBuffer，fc1 再执行一次只算 wgrad1。param_groups 两组（W2 存 dy，W1 存 dh），W2 那组的 #2 调用只在表里注明。
3. **compile、没有 fix**（`group-panel-compile-nofix`）：dense half 是一个 `CompiledFunctionBackward` 节点，InputBuffer 只有一个 slot 装 dy；**红框：dh 不在任何 InputBuffer 里**（编译图内部 buffer，调用完 del）。prehook 只截到 dy，param_groups 一组只有 dy；#1 整张跑（wgrad 算完丢掉），#2 拿 dy 整张再跑（dgrad2 重算、dgrad1 丢掉），共 8 个 GEMM。说明框用红色箭头指向红框。
4. **compile、有 fix**（`group-panel-compile-fix`）：节点 `_SplitBackwardFunctionBackward`，内含 dI 图（dgrad2 → dh → dgrad1）、`ctx.weight_live_ins`、dW 图；叶子前有 `ViewBackward0`。**橙框：dh 也不在 InputBuffer 里，而是由 dI 图交出、存进 ctx**。#1 只跑 dI 图，#2 的 dy 只用来叫起节点、只跑 dW 图，共 4 个 GEMM。说明框用橙色箭头指向橙框。

## 结构

- 分组（每张小图）：`group-panel-eager-nosplit`、`group-panel-eager-engine`、`group-panel-compile-nofix`、`group-panel-compile-fix`
- 每张小图内 id 前缀：`ens`（eager 不拆）、`es`（eager 拆）、`cn`（compile 没有 fix）、`cf`（compile 有 fix）
- 层：`group-<前缀>-graph`（K12 / K14 / K17）、`group-<前缀>-engine`（K3、K4）、`group-<前缀>-gt1` / `gt2`（K1、K6、K7、K9）、`group-es-timeline`（K1）
- autograd 节点：`node-ens-fc2`、`node-ens-fc1`、`node-es-fc2`、`node-es-fc1`（K12）；`node-cn-cfb`（K14、K15）；`node-cf-split`、`group-cf-di`、`group-cf-dw`（K17、K18）；`group-cf-leaves`（ViewBackward0，K19）
- InputBuffer：`node-<前缀>-ib-*`（K3、K4、K6）；`node-es-ib-fc1-w`（#2 新建，装同一个 dh，K6、K11）
- 核心框：`node-cn-dh-missing`（红框，K14、K15）、`node-cf-dh-ctx`（橙框，K18）；`node-cn-dh`（编译图内部 dh，K15）；`node-cf-ctx`（K18）
- param_groups：`node-es-param-groups`、`node-cn-param-groups`、`node-cf-param-groups`（K8、K10、K11）
- 说明框：`node-ens-callout`、`node-es-callout`、`node-cn-callout`、`node-cf-callout`
- 关键箭头：`edge-<前缀>-dgrad2-ib-fc1`（dh 存进 slot，K3）、`edge-<前缀>-ib-*-param-groups`（prehook，K5、K8）、`edge-<前缀>-param-groups-ib-*-w`（W 作为根梯度传入，K6、K9）、`edge-cn-callout-dh-missing`、`edge-cf-callout-dh-ctx`（说明框指向核心框）
- 形状：紫色直角粗框 = autograd 节点；节点里无边框圆角条 = 节点执行时的计算；椭圆 = 梯度 tensor；表格 = InputBuffer（一格一个 slot）；虚线框 = engine / GraphTask（运行时）；圆柱 = 运行时存储（ctx）
- 颜色：蓝 = I / dgrad，绿 = W / wgrad，橙 = I 交给 W 的 dh，红虚线 = 截不到、调用完就 del 的 dh，灰虚线 = prehook
- 阅读顺序：每张图从左边的 dy 开始 → GraphTask #1 的 InputBuffer → 节点 → 下一个 InputBuffer → prehook → param_groups → GraphTask #2；四张图从上到下对照看

## SVG 约定

- 尺寸：`viewBox="0 0 1780 3522"`（四张合一，单张宽 1780）；skill 默认渲染宽度 1240，本图小字多，建议渲染 2400
- id：方框 / 分组用 `node-*` / `group-*`，箭头 `edge-<from>-<to>`；只有关键图元有 id，普通文字和次要箭头没有
- `data-k="K1,K3"` 标在 `<g>` 分组上，表示这组图元画的是哪些知识
- 字体：`Noto Sans CJK SC`；没有外链、`<image>`、`<foreignObject>`
- 坐标是手排的：改一个框要检查有没有压到别的文字和连线
- 渲染方法：见 img-context-paste skill 的构建步骤第 5 步

## 修改记录

- 2026-10-27 首版：四张图（eager 不拆、eager 拆、compile 没有 fix、compile 有 fix）合一；关键图元加 id 和 data-k（基于 无，session 1e0800bb-a3d0-4288-8ecf-c5bad99f0310）

## 未解决

- [ ] K12：Linear 反向里的 `TBackward0` 没读 ATen 源码确认
- [ ] 后三张图右侧还有一列要点（"slot 号 = 边上的 input_nr"等），用户还没决定删不删
- [ ] eager 拆那张没画 W2 那组的 #2 GraphTask，只在 param_groups 表里注明
- [ ] 普通文字和次要箭头没有 id

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1780" height="3522" viewBox="0 0 1780 3522">
<defs>
<marker id="mb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#3b78d8"/></marker>
<marker id="mo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#e07b00"/></marker>
<marker id="mg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2e8b57"/></marker>
<marker id="mk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#888888"/></marker>
<marker id="md" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#444444"/></marker>
</defs>
<rect x="0" y="0" width="1780" height="3522" fill="white"/>
<g id="group-panel-eager-nosplit" transform="translate(0,0)">
<rect x="0" y="0" width="1780" height="712" rx="0" fill="white" stroke="none" stroke-width="1"/>
<text x="890.0" y="38" font-family="Noto Sans CJK SC" font-size="22" font-weight="bold" fill="#222" text-anchor="middle">eager、没有 split：一次 torch.autograd.backward，fc1 只执行一次</text>
<g id="group-ens-graph" data-k="K12,K13">
<rect x="220" y="66" width="1420" height="296" rx="6" fill="#fcfcfc" stroke="#cccccc" stroke-width="1.5"/>
<text x="234" y="90" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd 图　forward 时建好</text>
<g id="node-ens-fc2" data-k="K12">
<rect x="300" y="125" width="300" height="205" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="315" y="150" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">MmBackward0（fc2）</text>
<rect x="315" y="210" width="130" height="70" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="380.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad2</text>
<text x="380.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">h^T @ dy</text>
<rect x="460" y="210" width="130" height="70" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="525.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad2</text>
<text x="525.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dh = dy @ W2</text>
</g>
<g id="node-ens-fc1" data-k="K12">
<rect x="760" y="125" width="305" height="205" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="775" y="150" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">MmBackward0（fc1）</text>
<rect x="775" y="210" width="130" height="70" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="840.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad1</text>
<text x="840.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx = dh @ W1</text>
<rect x="920" y="210" width="130" height="70" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="985.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad1</text>
<text x="985.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">x^T @ dh</text>
</g>
<rect x="1190" y="88" width="160" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1270.0" y="113.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">TBackward0 (W2)</text>
<rect x="1430" y="88" width="190" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1525.0" y="113.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W2)</text>
<rect x="1190" y="152" width="190" height="56" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1285.0" y="177.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (x)</text>
<text x="1285.0" y="195.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">写进 x.grad</text>
<rect x="1190" y="255" width="160" height="46" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1270.0" y="283.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">TBackward0 (W1)</text>
<rect x="1430" y="255" width="190" height="46" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1525.0" y="283.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W1)</text>
<text x="1396" y="172" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">整次 backward 返回后，</text>
<text x="1396" y="189" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">stage 才把 x.grad 发给上一个 stage</text>
</g>
<g id="group-ens-engine" data-k="K3,K4">
<rect x="160" y="392" width="740" height="300" rx="8" fill="#fafafa" stroke="#888888" stroke-width="2" stroke-dasharray="9 6"/>
<text x="176" y="666" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd engine（运行时，不在 autograd 图里）</text>
<text x="176" y="685" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">InputBuffer 挂在 GraphTask 上，节点执行时被取走，用完即释放</text>
<g id="group-ens-gt1" data-k="K7,K13">
<rect x="180" y="410" width="700" height="200" rx="8" fill="#eef4ff" stroke="#3b78d8" stroke-width="1.8"/>
<g id="node-ens-ib-fc2" data-k="K3,K6">
<rect x="200" y="425" width="270" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="200" y1="455" x2="470" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="280" y1="455" x2="280" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="280.6" y="455.6" width="188.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="210" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">fc2 的 InputBuffer　num_inputs = 1</text>
<text x="240" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="375.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
</g>
<g id="node-ens-ib-fc1" data-k="K3,K4">
<rect x="560" y="425" width="280" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="560" y1="455" x2="840" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="640" y1="455" x2="640" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="640.6" y="455.6" width="198.8" height="33.8" rx="0" fill="#ffe3c2" stroke="none" stroke-width="1"/>
<text x="570" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">fc1 的 InputBuffer　num_inputs = 1</text>
<text x="600" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="740.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dh</text>
</g>
<text x="196" y="528" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1（唯一一个）　torch.autograd.backward(y, dy)</text>
<text x="196" y="551" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">exec_info：空 → 图里所有节点都执行，每个节点的所有输出都算</text>
<text x="196" y="571" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">dependencies_：fc1 = 1，fc2 跑完减到 0 → 就绪</text>
<text x="196" y="591" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#666" text-anchor="start">TBackward0 / AccumulateGrad 前面也各有一个 InputBuffer，图中省略</text>
</g>
<path d="M 200 636 L 865 636" fill="none" stroke="#444444" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#md)"/>
<text x="204" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1：fc2、fc1、TBackward0、AccumulateGrad 全部跑完才返回</text>
</g>
<ellipse cx="86" cy="472" rx="70" ry="32" fill="#e8e8e8" stroke="#333" stroke-width="1.2"/>
<text x="86" y="467" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
<text x="86" y="485" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">下一个 stage 送来的根梯度</text>
<path d="M 157 472 L 198 472" fill="none" stroke="#444444" stroke-width="2" stroke-linejoin="round" marker-end="url(#md)"/>
<path d="M 380 425 L 450 333" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="405" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="end">fc2 执行一次：dgrad2、wgrad2 都算</text>
<g id="edge-ens-dgrad2-ib-fc1" data-k="K3">
<path d="M 590 245 L 700 245 L 700 422" fill="none" stroke="#e07b00" stroke-width="2" stroke-linejoin="round" marker-end="url(#mo)"/>
<text x="606" y="284" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#e07b00" text-anchor="start">输出 dh</text>
<text x="606" y="302" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#e07b00" text-anchor="start">next_edge(0)</text>
<text x="606" y="318" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#e07b00" text-anchor="start">= (fc1, slot 0)</text>
</g>
<g id="edge-ens-ib-fc1-fc1" data-k="K7">
<path d="M 790 425 L 912 333" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="870" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="start">fc1 执行一次：dgrad1、wgrad1 都从这个 slot 拿 dh</text>
</g>
<path d="M 840 210 L 840 185 L 1187 185" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="1080" y="177" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">dx</text>
<path d="M 1050 245 L 1120 245 L 1120 278 L 1187 278" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<path d="M 1350 278 L 1427 278" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1376" y="270" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW1</text>
<path d="M 380 210 L 380 108 L 1187 108" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<path d="M 1350 108 L 1427 108" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1376" y="100" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW2</text>
<text x="700" y="102" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">同一次调用里就算</text>
<g id="node-ens-callout" data-k="K4,K13">
<rect x="930" y="430" width="520" height="122" rx="6" fill="#f6f6f6" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="942" y="453" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#222" text-anchor="start">没有 split：只有一个 GraphTask</text>
<text x="942" y="475" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">fc1 只执行一次，dgrad1 和 wgrad1 同时从这个 slot 拿到 dh</text>
<text x="942" y="497" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">fc1 跑完，InputBuffer 释放，dh 没人引用了 → 立刻释放</text>
<text x="942" y="519" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">不需要 prehook，也没有 param_groups</text>
<text x="942" y="541" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#b35c00" text-anchor="start">代价：dx 要等整次 backward（4 个 GEMM）跑完才发给上一个 stage</text>
</g></g>
<line x1="40" y1="727.0" x2="1740" y2="727.0" stroke="#bbbbbb" stroke-width="2" stroke-dasharray="10 8"/>
<g id="group-panel-eager-engine" transform="translate(0,742)">
<rect x="0" y="0" width="1780" height="880" rx="0" fill="white" stroke="none" stroke-width="1"/>
<text x="890.0" y="38" font-family="Noto Sans CJK SC" font-size="22" font-weight="bold" fill="#222" text-anchor="middle">eager、split 成 I / W：两次 autograd.grad，fc1 执行两次，dh 靠 param_groups 活到 W</text>
<g id="group-es-graph" data-k="K12">
<rect x="220" y="66" width="1420" height="296" rx="6" fill="#fcfcfc" stroke="#cccccc" stroke-width="1.5"/>
<text x="234" y="90" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd 图　forward 时建好，I、W 共用这一张图</text>
<g id="node-es-fc2" data-k="K12">
<rect x="300" y="125" width="300" height="205" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="315" y="150" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">MmBackward0（fc2）</text>
<rect x="315" y="210" width="130" height="70" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="380.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad2</text>
<text x="380.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">h^T @ dy</text>
<rect x="460" y="210" width="130" height="70" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="525.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad2</text>
<text x="525.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dh = dy @ W2</text>
</g>
<g id="node-es-fc1" data-k="K12">
<rect x="760" y="125" width="305" height="205" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="775" y="150" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">MmBackward0（fc1）</text>
<rect x="775" y="210" width="130" height="70" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="840.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad1</text>
<text x="840.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx = dh @ W1</text>
<rect x="920" y="210" width="130" height="70" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="985.0" y="242.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad1</text>
<text x="985.0" y="260.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">x^T @ dh</text>
</g>
<rect x="1190" y="88" width="160" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1270.0" y="113.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">TBackward0 (W2)</text>
<rect x="1430" y="88" width="190" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1525.0" y="113.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W2)</text>
<rect x="1190" y="152" width="190" height="56" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1285.0" y="177.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (x)</text>
<text x="1285.0" y="195.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx 在这里被截下</text>
<rect x="1190" y="255" width="160" height="46" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1270.0" y="283.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">TBackward0 (W1)</text>
<rect x="1430" y="255" width="190" height="46" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1525.0" y="283.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W1)</text>
</g>
<g id="group-es-engine" data-k="K3,K4">
<rect x="160" y="392" width="1100" height="300" rx="8" fill="#fafafa" stroke="#888888" stroke-width="2" stroke-dasharray="9 6"/>
<text x="176" y="666" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd engine（运行时，不在 autograd 图里）</text>
<text x="176" y="685" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">每次 autograd.grad 建一个 GraphTask；InputBuffer 挂在 GraphTask 上，节点执行时被取走，用完即释放</text>
<g id="group-es-gt1" data-k="K1,K7">
<rect x="180" y="410" width="680" height="200" rx="8" fill="#eef4ff" stroke="#3b78d8" stroke-width="1.8"/>
<g id="node-es-ib-fc2" data-k="K3">
<rect x="200" y="425" width="270" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="200" y1="455" x2="470" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="280" y1="455" x2="280" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="280.6" y="455.6" width="188.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="210" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">fc2 的 InputBuffer　num_inputs = 1</text>
<text x="240" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="375.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
</g>
<g id="node-es-ib-fc1" data-k="K3,K4,K5">
<rect x="560" y="425" width="280" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="560" y1="455" x2="840" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="640" y1="455" x2="640" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="640.6" y="455.6" width="198.8" height="33.8" rx="0" fill="#ffe3c2" stroke="none" stroke-width="1"/>
<text x="570" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">fc1 的 InputBuffer　num_inputs = 1</text>
<text x="600" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="740.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dh</text>
</g>
<text x="196" y="528" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1：I　autograd.grad(y → x)</text>
<text x="196" y="552" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">exec_info：执行 fc2、fc1；截取 AccumulateGrad(x)</text>
<text x="196" y="572" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">dependencies_：fc1 = 1，fc2 跑完减到 0 → 就绪</text>
</g>
<g id="group-es-gt2" data-k="K1,K6,K9">
<rect x="890" y="410" width="330" height="200" rx="8" fill="#eef8ee" stroke="#2e8b57" stroke-width="1.8"/>
<g id="node-es-ib-fc1-w" data-k="K6,K11">
<rect x="910" y="425" width="290" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="910" y1="455" x2="1200" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="990" y1="455" x2="990" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="990.6" y="455.6" width="208.8" height="33.8" rx="0" fill="#ffe3c2" stroke="none" stroke-width="1"/>
<text x="920" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">fc1 的 InputBuffer　这次调用新建的</text>
<text x="950" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="1095.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dh（同一个对象）</text>
</g>
<text x="906" y="528" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2：W（W1 那组）</text>
<text x="906" y="550" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">autograd.grad(root = fc1 + 存下的 dh → W1)</text>
<text x="906" y="570" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">exec_info：执行 fc1、TBackward0(W1)</text>
<text x="906" y="589" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">截取 AccumulateGrad(W1)</text>
</g>
</g>
<g id="node-es-param-groups" data-k="K8,K10,K11">
<text x="560" y="854" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">param_groups（stage_backward_input 返回，留到 W）</text>
<rect x="560" y="730" width="620" height="100" rx="0" fill="#fff3e6" stroke="#888888" stroke-width="1.3"/>
<line x1="560" y1="768" x2="1180" y2="768" stroke="#888" stroke-width="1.3"/>
<line x1="660" y1="730" x2="660" y2="830" stroke="#888" stroke-width="1.3"/>
<text x="610" y="754" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">W1 那组</text>
<text x="672" y="754" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">intermediate = fc1，grads = [dh 的引用]</text>
<text x="610" y="803" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">W2 那组</text>
<text x="672" y="791" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">intermediate = fc2，grads = [dy 的引用]</text>
<text x="672" y="813" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#666" text-anchor="start">fc2 的 prehook 存的；W 时同样另起一个 GraphTask，图中省略</text>
</g>
<ellipse cx="86" cy="472" rx="70" ry="32" fill="#e8e8e8" stroke="#333" stroke-width="1.2"/>
<text x="86" y="467" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
<text x="86" y="485" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">下一个 stage 送来的根梯度</text>
<path d="M 157 472 L 198 472" fill="none" stroke="#444444" stroke-width="2" stroke-linejoin="round" marker-end="url(#md)"/>
<path d="M 420 425 L 488 283" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="425" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="end">I：fc2 执行，只算 dgrad2</text>
<g id="edge-es-dgrad2-ib-fc1" data-k="K3">
<path d="M 590 245 L 700 245 L 700 422" fill="none" stroke="#e07b00" stroke-width="2" stroke-linejoin="round" marker-end="url(#mo)"/>
<text x="606" y="284" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#e07b00" text-anchor="start">输出 dh</text>
<text x="606" y="302" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#e07b00" text-anchor="start">next_edge(0)</text>
<text x="606" y="318" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#e07b00" text-anchor="start">= (fc1, slot 0)</text>
</g>
<path d="M 790 425 L 838 283" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="818" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="start">I：fc1 执行，只算 dgrad1</text>
<path d="M 840 210 L 840 185 L 1187 185" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="1072" y="177" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">dx → 上一个 stage</text>
<g id="edge-es-ib-fc1-param-groups" data-k="K5,K8,K11">
<path d="M 840 472 L 875 472 L 875 727" fill="none" stroke="#888888" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mk)"/>
<text x="884" y="714" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#666" text-anchor="start">prehook（fc1 执行前）多拿一个引用</text>
</g>
<g id="edge-es-param-groups-ib-fc1-w" data-k="K6,K9">
<path d="M 1180 749 L 1240 749 L 1240 472 L 1203 472" fill="none" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1250" y="718" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="start">W：作为根梯度传入</text>
</g>
<path d="M 985 425 L 985 283" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="995" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="start">W：fc1 再执行，只算 wgrad1</text>
<path d="M 1050 245 L 1120 245 L 1120 278 L 1187 278" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<path d="M 1350 278 L 1427 278" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1376" y="270" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW1</text>
<path d="M 380 210 L 380 108 L 1187 108" fill="none" stroke="#2e8b57" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#mg)"/>
<path d="M 1350 108 L 1427 108" fill="none" stroke="#2e8b57" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1376" y="100" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW2</text>
<text x="700" y="102" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">W2 那组的 W 才走这条（图中省略那次调用）</text>
<g id="group-es-timeline" data-k="K1">
<path d="M 200 636 L 1205 636" fill="none" stroke="#444444" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#md)"/>
<text x="204" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1 先跑完</text>
<text x="596" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#555" text-anchor="start">这段时间里跑其他 microbatch 的 F / I</text>
<text x="906" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2 后跑（W）</text>
</g>
<g id="node-es-callout" data-k="K1,K2,K11">
<rect x="165" y="712" width="382" height="150" rx="6" fill="#f6f6f6" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="177" y="734" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#222" text-anchor="start">为什么 prehook 要多拿一个引用</text>
<text x="177" y="753" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">I、W 拆成了两个 GraphTask（#1、#2），顺序执行</text>
<text x="177" y="772" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">同一个 dh 要先后放进两个 InputBuffer</text>
<text x="177" y="791" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">这两个 InputBuffer 不同时存在：#1 的随 fc1 执行完释放，</text>
<text x="177" y="810" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">#2 的到 W 才新建</text>
<text x="177" y="829" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">中间要有人拿着 dh 的引用，它才不会被释放</text>
<text x="177" y="853" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#b35c00" text-anchor="start">拿着它的就是 param_groups</text>
<path d="M 547 724 L 868 708" fill="none" stroke="#999" stroke-width="1.2" stroke-dasharray="2 3"/>
</g>
<text x="1300" y="440" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">slot 号 = 边上的 input_nr</text>
<text x="1300" y="466" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">同一个 slot 收到多个梯度时相加</text>
<text x="1300" y="492" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">一个 InputBuffer 只属于一次调用</text>
<text x="1300" y="518" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">dh 只有一份：两个 InputBuffer 装的是同一个对象</text></g>
<line x1="40" y1="1637.0" x2="1740" y2="1637.0" stroke="#bbbbbb" stroke-width="2" stroke-dasharray="10 8"/>
<g id="group-panel-compile-nofix" transform="translate(0,1652)">
<rect x="0" y="0" width="1780" height="910" rx="0" fill="white" stroke="none" stroke-width="1"/>
<text x="890" y="38" font-family="Noto Sans CJK SC" font-size="22" font-weight="bold" fill="#222" text-anchor="middle">torch.compile、split 成 I / W、没有 fix：整个 dense half 是一个节点，I、W 各把整张图跑一遍</text>
<g id="group-cn-graph" data-k="K14">
<rect x="220" y="66" width="1420" height="306" rx="6" fill="#fcfcfc" stroke="#cccccc" stroke-width="1.5"/>
<text x="234" y="90" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd 图　forward 时建好，I、W 共用这一张图</text>
<g id="node-cn-cfb" data-k="K14,K15">
<rect x="300" y="120" width="765" height="220" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="315" y="145" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">CompiledFunctionBackward</text>
<text x="315" y="164" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">整个 dense half 只有这一个 autograd 节点，里面是一张固定的编译图</text>
<rect x="315" y="205" width="130" height="65" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="380.0" y="234.5" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad2</text>
<text x="380.0" y="252.5" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dW2 = dy^T @ h</text>
<rect x="460" y="205" width="130" height="65" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="525.0" y="234.5" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad2</text>
<text x="525.0" y="252.5" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dh = dy @ W2</text>
<g id="node-cn-dh" data-k="K15">
<ellipse cx="682" cy="237" rx="70" ry="30" fill="#f8d7da" stroke="#aa3333" stroke-width="1.5" stroke-dasharray="5 3"/>
<text x="682" y="233" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dh</text>
<text x="682" y="250" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">bufN，调用完就 del</text>
</g>
<rect x="775" y="205" width="130" height="65" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="840.0" y="234.5" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad1</text>
<text x="840.0" y="252.5" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx = dh @ W1</text>
<rect x="920" y="205" width="130" height="65" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="985.0" y="234.5" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad1</text>
<text x="985.0" y="252.5" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dW1 = dh^T @ x</text>
<path d="M 590 237 L 609 237" fill="none" stroke="#aa3333" stroke-width="1.6" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mr)"/>
<path d="M 752 237 L 772 237" fill="none" stroke="#aa3333" stroke-width="1.6" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mr)"/>
<path d="M 682 267 L 682 300 L 985 300 L 985 273" fill="none" stroke="#aa3333" stroke-width="1.6" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mr)"/>
</g>
<rect x="1190" y="86" width="190" height="42" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1285.0" y="112.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W2)</text>
<rect x="1190" y="150" width="190" height="56" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1285.0" y="175.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (x)</text>
<text x="1285.0" y="193.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx 在这里被截下</text>
<rect x="1190" y="255" width="190" height="46" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1285.0" y="283.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W1)</text>
<path d="M 380 205 L 380 108 L 1187 108" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="700" y="101" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW1、dW2：#1 里也算了，但不在 exec_info → 丢掉；#2 里才截下</text>
<path d="M 840 205 L 840 185 L 1187 185" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="1090" y="177" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">dx</text>
<path d="M 1050 237 L 1120 237 L 1120 278 L 1187 278" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
</g>
<g id="group-cn-engine" data-k="K3,K4">
<rect x="160" y="392" width="1120" height="300" rx="8" fill="#fafafa" stroke="#888888" stroke-width="2" stroke-dasharray="9 6"/>
<text x="176" y="666" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd engine（运行时，不在 autograd 图里）</text>
<text x="176" y="685" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">每次 autograd.grad 建一个 GraphTask</text>
<g id="group-cn-gt1" data-k="K1,K7,K14">
<rect x="180" y="410" width="680" height="200" rx="8" fill="#eef4ff" stroke="#3b78d8" stroke-width="1.8"/>
<g id="node-cn-ib" data-k="K3,K14">
<rect x="200" y="425" width="360" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="200" y1="455" x2="560" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="280" y1="455" x2="280" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="280.6" y="455.6" width="278.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="210" y="446" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">CompiledFunctionBackward 的 InputBuffer</text>
<text x="240" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="420.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
</g>
<g id="node-cn-dh-missing" data-k="K14,K15">
<rect x="610" y="425" width="240" height="65" rx="4" fill="#fdecec" stroke="#aa3333" stroke-width="2.4" stroke-dasharray="6 3"/>
<text x="730" y="452" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#aa3333" text-anchor="middle">dh 不在任何 InputBuffer 里</text>
<text x="730" y="473" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#aa3333" text-anchor="middle">它是编译图内部的 buffer（bufN）</text>
</g>
<text x="196" y="530" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1：I　autograd.grad(y → x)</text>
<text x="196" y="553" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">exec_info：执行这个节点；截取 AccumulateGrad(x)</text>
<text x="196" y="573" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">节点执行 = 整张编译图跑完</text>
<text x="196" y="592" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">dW1、dW2 也算了，不在 exec_info → 丢掉</text>
</g>
<g id="group-cn-gt2" data-k="K1,K6,K9,K16">
<rect x="890" y="410" width="370" height="200" rx="8" fill="#eef8ee" stroke="#2e8b57" stroke-width="1.8"/>
<g id="node-cn-ib-w" data-k="K6,K11">
<rect x="905" y="425" width="340" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="905" y1="455" x2="1245" y2="455" stroke="#222" stroke-width="1.3"/>
<line x1="985" y1="455" x2="985" y2="490" stroke="#222" stroke-width="1.3"/>
<rect x="985.6" y="455.6" width="258.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="915" y="446" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#222" text-anchor="start">CompiledFunctionBackward 的 InputBuffer</text>
<text x="945" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="1115.0" y="478" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy（同一个对象）</text>
</g>
<text x="906" y="530" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2：W（唯一一组）</text>
<text x="906" y="552" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">root = 这个节点 + 存下的 dy → W1、W2</text>
<text x="906" y="572" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">截取 AccumulateGrad(W1)、AccumulateGrad(W2)</text>
<text x="906" y="592" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#aa3333" text-anchor="start">整张再跑：dgrad2 重算，dgrad1 算完丢掉</text>
</g>
<path d="M 200 636 L 1245 636" fill="none" stroke="#444444" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#md)"/>
<text x="204" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1 先跑完</text>
<text x="380" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#555" text-anchor="start">其间跑其他 microbatch 的 F / I</text>
<text x="906" y="628" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2 后跑（W）</text>
</g>
<ellipse cx="86" cy="472" rx="70" ry="32" fill="#e8e8e8" stroke="#333" stroke-width="1.2"/>
<text x="86" y="468" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
<text x="86" y="485" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">下一个 stage 送来的根梯度</text>
<path d="M 157 472 L 198 472" fill="none" stroke="#444444" stroke-width="2" stroke-linejoin="round" marker-end="url(#md)"/>
<path d="M 400 425 L 480 343" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="430" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="end">#1（I）：整张跑 4 个 GEMM</text>
<path d="M 1075 425 L 1000 343" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1050" y="382" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="start">#2（W）：整张再跑 4 个 GEMM</text>
<g id="node-cn-param-groups" data-k="K8,K10">
<rect x="165" y="730" width="535" height="62" rx="0" fill="#fff3e6" stroke="#888888" stroke-width="1.3"/>
<line x1="265" y1="730" x2="265" y2="792" stroke="#888" stroke-width="1.3"/>
<text x="215" y="766" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">唯一一组</text>
<text x="277" y="755" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">intermediate = CompiledFunctionBackward</text>
<text x="277" y="777" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">params = {W1, W2}，grads = [dy 的引用]</text>
<text x="165" y="816" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">param_groups（stage_backward_input 返回，留到 W）</text>
<text x="165" y="836" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#555" text-anchor="start">W1、W2 的路径都在这个节点交汇，所以只有一组</text>
</g>
<g id="edge-cn-ib-param-groups" data-k="K5,K8">
<path d="M 560 472 L 590 472 L 590 727" fill="none" stroke="#888888" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mk)"/>
<text x="578" y="714" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#666" text-anchor="end">prehook（节点执行前）多拿一个引用：只有 dy</text>
</g>
<g id="edge-cn-param-groups-ib-w" data-k="K6,K9">
<path d="M 680 792 L 680 880 L 1270 880 L 1270 472 L 1248 472" fill="none" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1265" y="900" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="end">W：作为根梯度传入（同一个 dy）</text>
</g>
<g id="node-cn-callout" data-k="K14,K15,K16">
<rect x="720" y="712" width="380" height="155" rx="6" fill="#f6f6f6" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="732" y="734" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#aa3333" text-anchor="start">为什么没有 fix 会重算：根子在上面这个红框</text>
<text x="732" y="753" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">dh 不在任何 InputBuffer 里，它是编译图内部的 buffer</text>
<text x="732" y="772" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">prehook 只能在 InputBuffer 上截梯度 → 截不到 dh</text>
<text x="732" y="791" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">param_groups 里只有 dy（节点只有这一个 slot）</text>
<text x="732" y="810" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">#2 手里没有 dh，只能拿 dy 把整张图再跑一遍</text>
<text x="732" y="829" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">dgrad2 重算；#1 里 wgrad 也照跑、算完丢掉</text>
<text x="732" y="848" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#b35c00" text-anchor="start">I、W 共 8 个 GEMM，只有 4 个有用</text>
</g>
<g id="edge-cn-callout-dh-missing" data-k="K14,K15,K16">
<path d="M 760 712 L 760 494" fill="none" stroke="#aa3333" stroke-width="2.2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mr)"/>
</g>
<text x="1310" y="440" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">slot 号 = 边上的 input_nr</text>
<text x="1310" y="466" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">这个节点 num_inputs = 1（只有 y）</text>
<text x="1310" y="492" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">InputBuffer 挂在 GraphTask 上，节点执行完就释放</text>
<text x="1310" y="518" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">engine 照样只标记要的边，但编译图没法只跑一部分</text>
<text x="1310" y="544" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">dh 在编译图里，调用一结束就 del，截不到</text></g>
<line x1="40" y1="2577.0" x2="1740" y2="2577.0" stroke="#bbbbbb" stroke-width="2" stroke-dasharray="10 8"/>
<g id="group-panel-compile-fix" transform="translate(0,2592)">
<rect x="0" y="0" width="1780" height="930" rx="0" fill="white" stroke="none" stroke-width="1"/>
<text x="890" y="38" font-family="Noto Sans CJK SC" font-size="22" font-weight="bold" fill="#222" text-anchor="middle">torch.compile、split 成 I / W、有 fix（SplitBackwardBackend）：I 只跑 dI 图，W 只跑 dW 图</text>
<g id="group-cf-graph" data-k="K17">
<rect x="220" y="66" width="1420" height="330" rx="6" fill="#fcfcfc" stroke="#cccccc" stroke-width="1.5"/>
<text x="234" y="90" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd 图　forward 时建好，I、W 共用这一张图</text>
<g id="node-cf-split" data-k="K17">
<rect x="300" y="115" width="765" height="253" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="315" y="138" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#222" text-anchor="start">_SplitBackwardFunctionBackward</text>
<text x="315" y="157" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">还是一个 autograd 节点，后面挂两张分开编译的图：dI 图、dW 图</text>
<g id="group-cf-di" data-k="K17,K18">
<rect x="318" y="168" width="440" height="76" rx="8" fill="none" stroke="#3b78d8" stroke-width="1.4" stroke-dasharray="6 4"/>
<text x="326" y="183" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#3b78d8" text-anchor="start">dI 图：#1 只跑它</text>
<rect x="330" y="190" width="120" height="48" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="390.0" y="211.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad2</text>
<text x="390.0" y="229.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dh = dy @ W2</text>
<ellipse cx="530" cy="214" rx="58" ry="24" fill="#ffe3c2" stroke="#e07b00" stroke-width="2"/>
<text x="530" y="210" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dh</text>
<text x="530" y="227" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">dI 图的输出</text>
<rect x="610" y="190" width="130" height="48" rx="12" fill="#cfe2ff" stroke="none" stroke-width="1"/>
<text x="675.0" y="211.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dgrad1</text>
<text x="675.0" y="229.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx = dh @ W1</text>
<path d="M 450 214 L 470 214" fill="none" stroke="#e07b00" stroke-width="2" stroke-linejoin="round" marker-end="url(#mo)"/>
<path d="M 588 214 L 608 214" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
</g>
<g id="node-cf-ctx" data-k="K18">
<path d="M 455 266 A 75.0 8 0 0 1 605 266 L 605 346 A 75.0 8 0 0 1 455 346 Z" fill="#fff3e6" stroke="#888" stroke-width="1.3" stroke-dasharray="5 3"/>
<path d="M 455 266 A 75.0 8 0 0 0 605 266" fill="none" stroke="#888" stroke-width="1.3" stroke-dasharray="5 3"/>
<text x="530.0" y="284" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#222" text-anchor="middle">ctx.weight_live_ins</text>
<text x="530.0" y="301" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#222" text-anchor="middle">dy, h, x, dh</text>
<text x="530.0" y="318" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#222" text-anchor="middle">detach 后存，活到 #2</text>
<path d="M 530 238 L 530 258" fill="none" stroke="#e07b00" stroke-width="2" stroke-linejoin="round" marker-end="url(#mo)"/>
</g>
<g id="group-cf-dw" data-k="K17,K18">
<rect x="625" y="252" width="175" height="110" rx="8" fill="none" stroke="#2e8b57" stroke-width="1.4" stroke-dasharray="6 4"/>
<text x="633" y="266" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#2e8b57" text-anchor="start">dW 图：#2 只跑它</text>
<rect x="635" y="272" width="155" height="40" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="712.5" y="289.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad2</text>
<text x="712.5" y="307.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dW2 = dy^T @ h</text>
<rect x="635" y="318" width="155" height="40" rx="12" fill="#c9ecc9" stroke="none" stroke-width="1"/>
<text x="712.5" y="335.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">wgrad1</text>
<text x="712.5" y="353.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dW1 = dh^T @ x</text>
<path d="M 605 292 L 633 292" fill="none" stroke="#e07b00" stroke-width="1.6" stroke-linejoin="round" marker-end="url(#mo)"/>
<path d="M 605 330 L 633 338" fill="none" stroke="#e07b00" stroke-width="1.6" stroke-linejoin="round" marker-end="url(#mo)"/>
</g>
</g>
<g id="group-cf-leaves" data-k="K19">
<rect x="1110" y="196" width="130" height="36" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1175.0" y="219.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">ViewBackward0</text>
<rect x="1310" y="188" width="200" height="52" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1410.0" y="211.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (x)</text>
<text x="1410.0" y="229.0" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="middle">dx 在这里被截下</text>
<path d="M 1240 214 L 1307 214" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<rect x="1110" y="272" width="130" height="36" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1175.0" y="295.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">ViewBackward0</text>
<rect x="1310" y="270" width="200" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1410.0" y="295.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W2)</text>
<path d="M 1240 290 L 1307 290" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<rect x="1110" y="318" width="130" height="36" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1175.0" y="341.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">ViewBackward0</text>
<rect x="1310" y="316" width="200" height="40" rx="0" fill="#f1eaf8" stroke="#6a3d9a" stroke-width="3"/>
<text x="1410.0" y="341.0" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">AccumulateGrad (W1)</text>
<path d="M 1240 336 L 1307 336" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
</g>
<path d="M 740 214 L 1107 214" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="1050" y="206" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#3b78d8" text-anchor="start">dx</text>
<path d="M 790 292 L 1107 290" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<path d="M 790 338 L 1107 336" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="960" y="285" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW2</text>
<text x="960" y="331" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#2e8b57" text-anchor="start">dW1</text>
</g>
<g id="group-cf-engine" data-k="K3,K4">
<rect x="160" y="410" width="1120" height="300" rx="8" fill="#fafafa" stroke="#888888" stroke-width="2" stroke-dasharray="9 6"/>
<text x="176" y="684" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">autograd engine（运行时，不在 autograd 图里）</text>
<text x="176" y="703" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#444" text-anchor="start">每次 autograd.grad 建一个 GraphTask</text>
<g id="group-cf-gt1" data-k="K1,K17">
<rect x="180" y="428" width="680" height="200" rx="8" fill="#eef4ff" stroke="#3b78d8" stroke-width="1.8"/>
<g id="node-cf-ib" data-k="K3">
<rect x="200" y="443" width="360" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="200" y1="473" x2="560" y2="473" stroke="#222" stroke-width="1.3"/>
<line x1="280" y1="473" x2="280" y2="508" stroke="#222" stroke-width="1.3"/>
<rect x="280.6" y="473.6" width="278.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="210" y="464" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#222" text-anchor="start">_SplitBackwardFunctionBackward 的 InputBuffer</text>
<text x="240" y="496" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="420.0" y="496" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
</g>
<g id="node-cf-dh-ctx" data-k="K18">
<rect x="610" y="443" width="240" height="65" rx="4" fill="#fff4e6" stroke="#e07b00" stroke-width="2.4" stroke-dasharray="6 3"/>
<text x="730" y="470" font-family="Noto Sans CJK SC" font-size="13" font-weight="bold" fill="#e07b00" text-anchor="middle">dh 也不在 InputBuffer 里</text>
<text x="730" y="491" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#e07b00" text-anchor="middle">dI 图把它交出来，存进节点的 ctx</text>
</g>
<text x="196" y="548" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1：I　autograd.grad(y → x)</text>
<text x="196" y="571" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">exec_info：执行这个节点；截取 AccumulateGrad(x)</text>
<text x="196" y="591" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">节点先问 engine 每个输入要不要</text>
<text x="196" y="610" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">只要 x 的 → 只跑 dI 图</text>
</g>
<g id="group-cf-gt2" data-k="K1,K6,K18,K20">
<rect x="890" y="428" width="370" height="200" rx="8" fill="#eef8ee" stroke="#2e8b57" stroke-width="1.8"/>
<g id="node-cf-ib-w" data-k="K6">
<rect x="905" y="443" width="340" height="65" rx="0" fill="white" stroke="#222" stroke-width="1.3"/>
<line x1="905" y1="473" x2="1245" y2="473" stroke="#222" stroke-width="1.3"/>
<line x1="985" y1="473" x2="985" y2="508" stroke="#222" stroke-width="1.3"/>
<rect x="985.6" y="473.6" width="258.8" height="33.8" rx="0" fill="#e8e8e8" stroke="none" stroke-width="1"/>
<text x="915" y="464" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#222" text-anchor="start">_SplitBackwardFunctionBackward 的 InputBuffer</text>
<text x="945" y="496" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">slot 0</text>
<text x="1115.0" y="496" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy（同一个对象）</text>
</g>
<text x="906" y="548" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2：W（唯一一组）</text>
<text x="906" y="570" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">root = 这个节点 + 存下的 dy → W1、W2</text>
<text x="906" y="590" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#333" text-anchor="start">dy 只用来把节点叫起来</text>
<text x="906" y="610" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="start">只跑 dW 图，数据从 ctx 拿</text>
</g>
<path d="M 200 654 L 1245 654" fill="none" stroke="#444444" stroke-width="1.4" stroke-linejoin="round" marker-end="url(#md)"/>
<text x="204" y="646" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1f4f9e" text-anchor="start">GraphTask #1 先跑完</text>
<text x="380" y="646" font-family="Noto Sans CJK SC" font-size="11" font-weight="normal" fill="#555" text-anchor="start">其间跑其他 microbatch 的 F / I</text>
<text x="906" y="646" font-family="Noto Sans CJK SC" font-size="11" font-weight="bold" fill="#1d6b41" text-anchor="start">GraphTask #2 后跑（W）</text>
</g>
<ellipse cx="86" cy="490" rx="70" ry="32" fill="#e8e8e8" stroke="#333" stroke-width="1.2"/>
<text x="86" y="486" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">dy</text>
<text x="86" y="503" font-family="Noto Sans CJK SC" font-size="10" font-weight="normal" fill="#222" text-anchor="middle">下一个 stage 送来的根梯度</text>
<path d="M 157 490 L 198 490" fill="none" stroke="#444444" stroke-width="2" stroke-linejoin="round" marker-end="url(#md)"/>
<path d="M 390 443 L 390 241" fill="none" stroke="#3b78d8" stroke-width="2" stroke-linejoin="round" marker-end="url(#mb)"/>
<text x="400" y="424" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#3b78d8" text-anchor="start">#1（I）：只跑 dI 图</text>
<path d="M 1075 443 L 1075 382 L 712 382 L 712 365" fill="none" stroke="#2e8b57" stroke-width="2" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1085" y="424" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="start">#2（W）：只跑 dW 图</text>
<g id="node-cf-param-groups" data-k="K8,K10">
<rect x="165" y="748" width="535" height="62" rx="0" fill="#fff3e6" stroke="#888888" stroke-width="1.3"/>
<line x1="265" y1="748" x2="265" y2="810" stroke="#888" stroke-width="1.3"/>
<text x="215" y="784" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="middle">唯一一组</text>
<text x="277" y="773" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">intermediate = _SplitBackwardFunctionBackward</text>
<text x="277" y="795" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">params = {W1, W2}，grads = [dy 的引用]</text>
<text x="165" y="834" font-family="Noto Sans CJK SC" font-size="14" font-weight="bold" fill="#222" text-anchor="start">param_groups（stage_backward_input 返回，留到 W）</text>
<text x="165" y="854" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#555" text-anchor="start">只有 dy：#2 只用它把节点叫起来，不靠它算</text>
</g>
<g id="edge-cf-ib-param-groups" data-k="K5,K8">
<path d="M 560 490 L 590 490 L 590 745" fill="none" stroke="#888888" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mk)"/>
<text x="578" y="732" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#666" text-anchor="end">prehook（节点执行前）多拿一个引用：只有 dy</text>
</g>
<g id="edge-cf-param-groups-ib-w" data-k="K6,K9">
<path d="M 680 810 L 680 898 L 1270 898 L 1270 490 L 1248 490" fill="none" stroke="#2e8b57" stroke-width="2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mg)"/>
<text x="1265" y="918" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#2e8b57" text-anchor="end">W：作为根梯度传入（同一个 dy）</text>
</g>
<g id="node-cf-callout" data-k="K18,K20">
<rect x="720" y="730" width="380" height="136" rx="6" fill="#f6f6f6" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="732" y="752" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#e07b00" text-anchor="start">有 fix 之后：dh 还是不在 InputBuffer 里（上面橙框）</text>
<text x="732" y="771" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">但不再需要 InputBuffer + prehook 来保住它</text>
<text x="732" y="790" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">dI 图把 dh 作为输出交出来，detach 后存进节点的 ctx</text>
<text x="732" y="809" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">param_groups 里照样只有 dy，只用来在 #2 叫起节点</text>
<text x="732" y="828" font-family="Noto Sans CJK SC" font-size="12" font-weight="normal" fill="#222" text-anchor="start">#1 只跑 dI 图；#2 只跑 dW 图，数据从 ctx 拿</text>
<text x="732" y="847" font-family="Noto Sans CJK SC" font-size="12" font-weight="bold" fill="#b35c00" text-anchor="start">I、W 共 4 个 GEMM，和 eager split 一样</text>
</g>
<g id="edge-cf-callout-dh-ctx" data-k="K18">
<path d="M 760 730 L 760 512" fill="none" stroke="#e07b00" stroke-width="2.2" stroke-dasharray="7 5" stroke-linejoin="round" marker-end="url(#mo)"/>
</g>
<text x="1310" y="458" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">slot 号 = 边上的 input_nr</text>
<text x="1310" y="484" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">这个节点 num_inputs = 1（只有 y）</text>
<text x="1310" y="510" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">节点用 _will_engine_execute_node 问每个输入要不要</text>
<text x="1310" y="536" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">ViewBackward0：实现里给叶子包了一层 view_as</text>
<text x="1310" y="562" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">（engine 不肯回答被 autograd.grad 截取的叶子）</text>
<text x="1310" y="588" font-family="Noto Sans CJK SC" font-size="13" font-weight="normal" fill="#222" text-anchor="start">dh 只有一份：存在 ctx 里活到 #2</text></g>
</svg>
```
