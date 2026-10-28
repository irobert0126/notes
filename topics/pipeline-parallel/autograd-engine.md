---
title: Autograd 引擎：GraphTask、InputBuffer 与 prehook
topic: pipeline-parallel
tags: [autograd, GraphTask, InputBuffer, prehook, param_groups, AccumulateGrad, CompiledFunctionBackward]
related: [topics/pipeline-parallel/zero-bubble-backward-split.md]
sources: [sources/backward_split_context.md]
---

# Autograd 引擎：GraphTask、InputBuffer 与 prehook

## 一句话目的

理解 backward 被 engine 执行的机制，这是 zero-bubble 的 dI / dW 拆分能在 eager 下工作、却在 compile 下失效的原因。

## 核心知识点

- **GraphTask**：engine 为每次 `autograd.grad` 或 `backward` 调用建立的执行上下文，持有 exec_info、依赖计数和各节点的 InputBuffer。
- **InputBuffer / slot**：节点执行前收梯度的槽。slot 个数等于节点的 num_inputs，slot 号是边上的 input_nr。同一个 slot 收到多个梯度时相加；节点执行时被取走，用完即释放。
- **exec_info**：这次要执行哪些节点、截取哪些边。`backward()` 不带 inputs 时全部执行；`autograd.grad(inputs=...)` 只执行被标记的节点。
- **prehook**：用 `Node.register_prehook` 挂在节点上，节点执行前可以拿到 InputBuffer 里的梯度。
- **param_groups**：`stage_backward_input` 返回的分组，记录权重、交汇节点和 prehook 截到的梯度引用，供 W 阶段使用。
- **GradientEdge**：W 阶段 `autograd.grad` 的起点，即（中间节点，输入编号），配合保存下来的梯度作为根梯度。
- **AccumulateGrad**：叶子 tensor 的梯度汇入节点；被 `autograd.grad` 截取时不会累加到 `.grad`。
- **CompiledFunctionBackward**：AOTAutograd 把整个编译后的 backward 包成的单个节点，每次调用整张编译图都会跑完。

## 注意

- dI 和 dW 不是两个独立的 autograd 节点，而是每个 `MmBackward0` 的两个输出；compile 合并的是按 op 划分的节点。
- InputBuffer 只在 I 那次调用中存在；真正跨两次 GraphTask 保存 dh 的是 param_groups 里的引用。

## 来源

- [完整上下文](../../sources/backward_split_context.md)
