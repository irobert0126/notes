---
title: VMS 修复 fragment
topic: gpu-memory
tags: [VMS, VMM, fragment, expandable_segments, GMLake, 空洞]
related: [topics/gpu-memory/vmm-expandable-segments.md]
sources: [sources/vms-fragment-repair.md]
---

# VMS 修复 fragment

## 一句话目的

用实际例子解释 VMS 是怎么解决 fragment 的。

## 核心知识点（已润色）

用一个实际在 VA 上产生 fragment 的例子：VMS 把 fragment 区域（空洞）对应的 chunk 重新映射到段尾新的连续 VA 上，让这些 chunk 被使用，从而变相解决了"fragment 区域占着 chunk"的问题。VA 内部的 fragment 仍然存在，但 VA 可以比 HBM 大（预留 VA 不占显存，只有映射上的 chunk 占显存），这些 fragment 本质上不影响 HBM 的利用率。

fragment 的根源是"连续 VA 的要求"和"物理 chunk 不连续"之间的矛盾；VMS 不挪动已有的 chunk，而是再用 VMM 映射一次，绕开这个矛盾。

![VMS 修复 fragment 的四步示意](../../images/vms-fragment-repair.png)

## 来源

- [完整上下文（SVG 源码）](../../sources/vms-fragment-repair.md)
- 对应论文：GMLake（arXiv 2401.08156，ASPLOS'24）
