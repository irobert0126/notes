# 七个 CUDA 驱动内存操作对 HBM、VA、段的影响（driver_ops_impact）

新 session 只需读本 paste。

- 基于：无（首版）
- 对应文件：`driver_ops_impact.svg`（源文件，内容同第 8 节）

## 核心论点

**一句话目的：** 逐个列出七个底层 CUDA 驱动调用对 HBM、VA 和段的影响，弄清哪些只动物理内存，哪些动 VA 映射。

**核心知识点（草稿，待你确认）：** 每种 CUDA 驱动调用对应一个确定的操作：

- cuMemCreate：分配一个 2 MB 物理句柄，HBM 占用增加；不涉及 VA。
- cuMemRelease：释放一个物理句柄，HBM 归还 driver，reserved 减少；代码里它在解除映射之后调用（CUDA 文档未核对）。
- cuMemAddressReserve：在 VA 上预留一段地址范围；不映射，不占 HBM。
- cuMemAddressFree：归还一段预留的 VA 范围；代码里它在解除映射之后调用。
- cuMemMap：把一个物理句柄映射到 VA 的某个位置，建立一条 VA → chunk 的映射。
- cuMemSetAccess：给这条映射设置读写访问权限。
- cuMemUnmap：删除 VA 上的一条映射；HBM 不释放，chunk 仍在。

只有 cuMemRelease 会让 reserved 下降。PyTorch 的 expandable segments 也用这七个调用，但它的解除映射（unmapHandles）会同时 cuMemRelease，所以也归还 HBM。

**整张图的精髓：** 七个调用各做一件事。cuMemCreate / cuMemRelease 管物理句柄，其余五个管 VA 和映射。

**措辞说明：** 方框里的 chunk 编号、VA 格子是示意。每个方框的状态变化是按代码路径手推的，未运行。"段（SM）"按 VmmSegment 理解，待确认。

## 术语

| 术语 | 含义 |
|---|---|
| 物理句柄（chunk） | cuMemCreate 分配的 2 MB 物理内存（HBM） |
| VA 预留 | cuMemAddressReserve 占下的一段地址范围，还没有映射 |
| 映射 | 把一个物理句柄放到 VA 的某个位置，cuMemMap 建立，cuMemUnmap 删除 |
| 访问权限 | 映射上的读写设置，由 cuMemSetAccess 设置 |
| reserved | 分配器持有的物理内存；只有 cuMemRelease 能让它下降 |
| 段（VmmSegment） | 一段 VA 与它映射的 chunk 列表；图里的 SM 按这个理解（待确认） |
| 驱动封装 | PyTorch 的 DriverAPI（函数指针，dlsym 取得）或 cudaMalloc 的包装 |

## 背景

- 起因：要弄清楚七个底层内存调用各自改了什么，尤其是哪些会降低 reserved。
- 来源：GMLake patch 的 cuda_vmm_allocator.h 和 CUDACachingAllocator.cpp；PyTorch 的 fbsource 版本（CodeHub 查看）。
- 限制：本次没有联网，CUDA 驱动文档没有核对。

## 知识

| ID | 内容 | 来源 | 状态 | 确认 | 你的理解 | 对应图元 |
|---|---|---|---|---|---|---|
| K1 | cuMemCreate 只分配一个 2 MB 物理句柄（HBM 占用增加），不改 VA 映射。 | GMLake cuda_vmm_allocator.h :236（PhyBlock 构造） | verified（读过代码） | 待确认 |  | 2 号方框（cuMemCreate） |
| K2 | cuMemRelease 只释放物理句柄，HBM 归还 driver。它是七个调用里唯一会让 reserved 下降的。 | GMLake h:243、h:248；PyTorch CUDACachingAllocator.cpp :1038 | verified（读过代码） | 待确认 |  | 2 号方框（cuMemRelease） |
| K3 | cuMemAddressReserve 只在 VA 上预留一段地址范围，不映射，不占 HBM。 | GMLake h:279；PyTorch :536 | verified（读过代码） | 待确认 |  | 3 号方框 |
| K4 | cuMemAddressFree 归还预留的 VA 范围。代码里它总是在解除映射之后调用（析构路径）。 | GMLake h:306（析构），h:285（构造失败，此时还没映射）；PyTorch :923–929（先 unmapHandles，后 cuMemAddressFree） | verified（代码顺序） | 待确认 |  | 4 号方框（cuMemAddressFree） |
| K5 | cuMemMap 把一个 chunk 映射到 VA 的某个位置。GMLake 每个 VirBlock 构造时调用一次。 | GMLake h:351；PyTorch :983 | verified（读过代码） | 待确认 |  | 5 号方框（cuMemMap） |
| K6 | cuMemSetAccess 给映射设置访问权限；映射建立后要设置，才能真正读写。GMLake 通过 setMemAccess 调用，VirBlock 构造时执行。 | GMLake h:185、h:352；PyTorch :963 | verified（读过代码） | 待确认 |  | 6 号方框（cuMemSetAccess） |
| K7 | cuMemUnmap 删除 VA 上的映射，HBM 不释放，chunk 仍在。GMLake 只在 VirDevPtr 析构时调用。 | GMLake h:304；PyTorch :1022（unmapHandles） | verified（读过代码） | 待确认 |  | 7 号方框（cuMemUnmap） |
| K8 | 整个 GMLake patch 里 cuMemUnmap 只有一个调用点：VirDevPtr::release_resources。 | grep 整个 patch（两个文件） | verified（grep） | 待确认 |  | 7 号方框 |
| K9 | PyTorch 的 expandable segments 也用了这七个 driver 调用。拼接、碎片池、页回收的逻辑在 PyTorch 里没有找到。 | fbsource c10/cuda/CUDACachingAllocator.cpp（CodeHub grep） | verified（grep，fbsource 版本） | 待确认 |  | 整张图的基线 |
| K10 | PyTorch 的 unmapHandles 先 cuMemUnmap，再 cuMemRelease，所以它的解除映射同时归还 HBM。GMLake 的 unmap 不归还 HBM。两边在这一点上不同。 | PyTorch :1022、:1038；GMLake h:304、h:243（PhyBlock 析构时才 release） | verified（读过代码） | 待确认 |  | 7 号方框的说明 |
| K11 | PyTorch 的 DriverAPI 用 dlsym 从 libcuda 取得 cuMem* 函数指针，所以 cuMem* 调用在源码里是显式的，但指针是运行时才确定的。 | fbsource driver_api.cpp :67、:75 | verified（读过代码） | 待确认 |  | 分层图第 3 层 |
| K12 | 调用层级：算子 → 分配器 → 驱动封装 → runtime（cudaMalloc、cudaFree）或直接 driver（cuMem*）→ 内核驱动和 MMU。cuMem* 不经过 runtime。 | 分层图的结构（代码路径）；内核驱动和 MMU 这一层是通用知识 | verified（前四层）；unverified（第 5 层） | 待确认 |  | 分层图 |
| K13 | cuMemAddressFree 之前要求映射已经解除。CUDA 文档我没有核对，这一点只能从两边代码的顺序看出来。 | GMLake h:304–306；PyTorch :925–929 | unverified（CUDA 文档未核对，本次无联网） | 待确认 |  | 4 号方框的前提 |
| K14 | 如果在映射还在的时候直接调用 cuMemAddressFree，驱动会怎么处理（报错、残留映射，还是别的），我没有验证。 | 无；需要 CUDA 文档或在 GPU 上实测 | unverified | 待确认 |  | 4 号方框的"调用前"假设 |
| K15 | 图里七个方框的状态变化是按代码路径手推的，没有运行验证。 | 手推 | inferred（手推） | 待确认 |  | 全图 |
| K16 | 图里的"段（SM）"按 VmmSegment 理解。用户原话是 SM，尚未确认。 | 图的说明 | inferred（命名） | 待确认 |  | 全图的段行 |

状态：verified（读过代码，或 grep 过）/ unverified（通用知识，或 CUDA 文档未核对）/ inferred（手推的例子，未运行）。确认：待确认 / 已确认。

## 分组讲解（待确认）

### 第 1 组：哪些调用动 VA，哪些只动 HBM（K1、K3、K4、K5、K6、K7）

- **只动 HBM 的两个**：cuMemCreate 分配一个物理句柄；cuMemRelease 释放它（K1、K2）。
- **动 VA 的五个**：cuMemAddressReserve 预留；cuMemMap 映射；cuMemSetAccess 设访问权限；cuMemUnmap 解除映射；cuMemAddressFree 归还。预留和映射是两件事：预留不占 HBM，映射才让 VA 指向 chunk（K3、K5、K6、K7）。

检查题：cuMemCreate 之后，立即 cuMemMap 的话，HBM 和 VA 各变了几次？cuMemRelease 之前必须做什么？

### 第 2 组：reserved 和 HBM 的关系（K2、K7、K10）

- reserved 只随物理句柄的分配和释放变化。cuMemUnmap 删映射，不改 reserved。
- PyTorch 的 unmapHandles 先 cuMemUnmap 再 cuMemRelease，所以它的解除映射会降低 reserved（K10）。GMLake 的 VirDevPtr 析构只做 unmap，PhyBlock 在没有段持有时才析构，才会 cuMemRelease。

检查题：GMLake 里，什么时候 unmap 之后 HBM 还会释放？为什么两边的 unmap 不一样？

### 第 3 组：调用顺序和前提（K4、K11、K12、K13、K14）

- 代码里 cuMemAddressFree 总是在解除映射之后调用（K4）。
- CUDA 驱动是否要求这个顺序，我没有核对（K13、K14）。
- PyTorch 的 cuMem* 调用是显式的，函数指针运行时用 dlsym 取得（K11）。

检查题：如果在映射还在时调用 cuMemAddressFree，驱动会怎么处理？这一点需要查 CUDA 文档或实测确认。

## 图的内容

- 七个方框，每个对应一个驱动调用：cuMemCreate、cuMemRelease、cuMemAddressReserve、cuMemAddressFree、cuMemMap、cuMemSetAccess、cuMemUnmap。
- 每个方框上半是调用前，下半是调用后。每半有 VA 格子（上）、HBM chunk（下），映射用箭头连接。
- 段的变化写在每半下面的文字里。
- 第 8 格是颜色图例。

## 结构

- 布局：4 列 × 2 行，前七格是七个调用，第 8 格是图例。
- 颜色：绿色格子 = 已映射且可读写；橙色虚线 = 已映射但没有访问权限；空心 = 预留，没有映射；虚线空框 = 没有 VA 或没有 chunk。
- HBM 紫色方框 = 有 chunk；虚线空框 = 没有。

## SVG 约定

- 尺寸：`viewBox="0 0 1240 770"`
- 字体：`'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial`
- chunk 编号和 VA 格子都是示意。

## SVG 源码

本版完整 SVG，修改时从这里取出。

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="770" viewBox="0 0 1240 770" font-family="'PingFang SC','Hiragino Sans GB','Helvetica Neue',Arial,sans-serif">
<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#34495e"/></marker></defs>
<rect width="1240" height="770" fill="#ffffff"/>
<text x="10.0" y="28.0" font-size="16" font-weight="bold" fill="#1c2833">七个底层内存操作对 HBM 和 VA 的影响（箭头 = 映射）</text>
<text x="10.0" y="50.0" font-size="11" fill="#566573">每个方框是一个 CUDA 驱动调用。方框内上半 = 调用前，下半 = 调用后。</text>
<rect x="10.0" y="84.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="20.0" y="102.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemCreate</text>
<text x="20.0" y="115.0" font-size="8.5" fill="#566573">调用点：PhyBlock 构造（h:236）</text>
<text x="20.0" y="129.0" font-size="8.5" fill="#33414e">新建一个 2 MB 物理句柄。这时还没有 VA 映射它，HBM 占用增加。</text>
<text x="18.0" y="156.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="18.0" y="176.0" font-size="8" fill="#566573">VA</text>
<text x="18.0" y="220.0" font-size="8" fill="#566573">HBM</text>
<rect x="50.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="78.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="106.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="134.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="50.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="65.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H0</text>
<rect x="90.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="105.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H1</text>
<rect x="130.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="145.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="170.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="185.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="20.0" y="240.0" font-size="8.5" fill="#1c2833">没有 chunk；段里没有 PhyBlock</text>
<line x1="18.0" y1="262.0" x2="292.0" y2="262.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="18.0" y="278.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="18.0" y="298.0" font-size="8" fill="#566573">VA</text>
<text x="18.0" y="342.0" font-size="8" fill="#566573">HBM</text>
<rect x="50.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="78.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="106.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="134.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="50.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="65.0" y="341.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="90.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="105.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H1</text>
<rect x="130.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="145.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="170.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="185.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="20.0" y="362.0" font-size="8.5" fill="#1c2833">+1 chunk（HBM +2 MB）；PhyBlock 建立，之后由段持有</text>
<text x="20.0" y="392.0" font-size="10" font-weight="bold" fill="#ca6f1e">reserved +2 MB</text>
<rect x="312.0" y="84.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="322.0" y="102.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemRelease</text>
<text x="322.0" y="115.0" font-size="8.5" fill="#566573">调用点：PhyBlock 析构（h:243）</text>
<text x="322.0" y="129.0" font-size="8.5" fill="#33414e">释放物理句柄，HBM 归还 driver。只在没有段再持有 PhyBlock 时调</text>
<text x="322.0" y="140.0" font-size="8.5" fill="#33414e">用。</text>
<text x="320.0" y="156.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="320.0" y="176.0" font-size="8" fill="#566573">VA</text>
<text x="320.0" y="220.0" font-size="8" fill="#566573">HBM</text>
<rect x="352.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="380.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="408.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="436.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="352.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="367.0" y="219.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="392.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="407.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H1</text>
<rect x="432.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="447.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="472.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="487.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="322.0" y="240.0" font-size="8.5" fill="#1c2833">H0 只剩句柄，没有映射；段还持有它</text>
<line x1="320.0" y1="262.0" x2="594.0" y2="262.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="320.0" y="278.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="320.0" y="298.0" font-size="8" fill="#566573">VA</text>
<text x="320.0" y="342.0" font-size="8" fill="#566573">HBM</text>
<rect x="352.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="380.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="408.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="436.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="352.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="367.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H0</text>
<rect x="392.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="407.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H1</text>
<rect x="432.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="447.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="472.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="487.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="322.0" y="362.0" font-size="8.5" fill="#1c2833">H0 归还 driver；PhyBlock 对象销毁</text>
<text x="322.0" y="392.0" font-size="10" font-weight="bold" fill="#ca6f1e">reserved −2 MB</text>
<rect x="614.0" y="84.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="624.0" y="102.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemAddressReserve</text>
<text x="624.0" y="115.0" font-size="8.5" fill="#566573">调用点：VirDevPtr 构造（h:279）</text>
<text x="624.0" y="129.0" font-size="8.5" fill="#33414e">预留一段 VA，还没有映射。VA 不占 HBM。</text>
<text x="622.0" y="156.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="622.0" y="176.0" font-size="8" fill="#566573">VA</text>
<text x="622.0" y="220.0" font-size="8" fill="#566573">HBM</text>
<rect x="654.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="682.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="710.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="738.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="654.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="669.0" y="219.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="694.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="709.0" y="219.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="734.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="749.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="774.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="789.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="624.0" y="240.0" font-size="8.5" fill="#1c2833">没有 VA 段</text>
<line x1="622.0" y1="262.0" x2="896.0" y2="262.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="622.0" y="278.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="622.0" y="298.0" font-size="8" fill="#566573">VA</text>
<text x="622.0" y="342.0" font-size="8" fill="#566573">HBM</text>
<rect x="654.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="682.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="710.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="738.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="654.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="669.0" y="341.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="694.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="709.0" y="341.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="734.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="749.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="774.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="789.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="624.0" y="362.0" font-size="8.5" fill="#1c2833">+4 格 VA 预留（空）；VirDevPtr 建立</text>
<text x="624.0" y="392.0" font-size="10" font-weight="bold" fill="#566573">reserved 不变</text>
<rect x="916.0" y="84.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="926.0" y="102.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemAddressFree</text>
<text x="926.0" y="115.0" font-size="8.5" fill="#566573">调用点：VirDevPtr 析构（h:306）；构造失败（h:285）</text>
<text x="926.0" y="129.0" font-size="8.5" fill="#33414e">归还预留的 VA 范围。正常路径在析构时调用，构造失败时也调用来清</text>
<text x="926.0" y="140.0" font-size="8.5" fill="#33414e">理。</text>
<text x="924.0" y="156.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="924.0" y="176.0" font-size="8" fill="#566573">VA</text>
<text x="924.0" y="220.0" font-size="8" fill="#566573">HBM</text>
<rect x="956.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="984.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="1012.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="1040.0" y="162.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="956.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="971.0" y="219.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="996.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="1011.0" y="219.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="1036.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="1051.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="1076.0" y="206.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="1091.0" y="219.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="926.0" y="240.0" font-size="8.5" fill="#1c2833">VA 预留，没有映射</text>
<line x1="924.0" y1="262.0" x2="1198.0" y2="262.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="924.0" y="278.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="924.0" y="298.0" font-size="8" fill="#566573">VA</text>
<text x="924.0" y="342.0" font-size="8" fill="#566573">HBM</text>
<rect x="956.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="984.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="1012.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="1040.0" y="284.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<rect x="956.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="971.0" y="341.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="996.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="1011.0" y="341.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="1036.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="1051.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="1076.0" y="328.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="1091.0" y="341.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="926.0" y="362.0" font-size="8.5" fill="#1c2833">VA 范围归还；HBM 不动</text>
<text x="926.0" y="392.0" font-size="10" font-weight="bold" fill="#566573">reserved 不变</text>
<rect x="10.0" y="414.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="20.0" y="432.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemMap</text>
<text x="20.0" y="445.0" font-size="8.5" fill="#566573">调用点：VirBlock 构造（h:351）</text>
<text x="20.0" y="459.0" font-size="8.5" fill="#33414e">把一个 chunk 映射到 VA 的某个位置。VirBlock 构造时每个 chunk</text>
<text x="20.0" y="470.0" font-size="8.5" fill="#33414e"> 调用一次。</text>
<text x="18.0" y="486.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="18.0" y="506.0" font-size="8" fill="#566573">VA</text>
<text x="18.0" y="550.0" font-size="8" fill="#566573">HBM</text>
<rect x="50.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="78.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="106.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="134.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="50.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="65.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="90.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="105.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="130.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="145.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="170.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="185.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="20.0" y="570.0" font-size="8.5" fill="#1c2833">VA 预留，没有映射</text>
<line x1="18.0" y1="592.0" x2="292.0" y2="592.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="18.0" y="608.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="18.0" y="628.0" font-size="8" fill="#566573">VA</text>
<text x="18.0" y="672.0" font-size="8" fill="#566573">HBM</text>
<rect x="50.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="63.0" y="627.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">H0</text>
<rect x="78.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="106.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="134.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="50.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="65.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="90.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="105.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="130.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="145.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="170.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="185.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<line x1="63.0" y1="634.0" x2="65.0" y2="657.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="20.0" y="692.0" font-size="8.5" fill="#1c2833">格 0 映射到 H0；段的 vir_blocks +1（chunk 本身不变）</text>
<text x="20.0" y="722.0" font-size="10" font-weight="bold" fill="#566573">reserved 不变</text>
<rect x="312.0" y="414.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="322.0" y="432.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemSetAccess</text>
<text x="322.0" y="445.0" font-size="8.5" fill="#566573">调用点：setMemAccess（h:185），VirBlock 构造调用（h:352）</text>
<text x="322.0" y="459.0" font-size="8.5" fill="#33414e">给映射设置读写访问权限。映射建立后还要设置，才能真正访问。</text>
<text x="320.0" y="486.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="320.0" y="506.0" font-size="8" fill="#566573">VA</text>
<text x="320.0" y="550.0" font-size="8" fill="#566573">HBM</text>
<rect x="352.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.1" stroke-dasharray="3 2"/>
<text x="365.0" y="505.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ca6f1e">H0</text>
<rect x="380.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="408.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="436.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="352.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="367.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="392.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="407.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="432.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="447.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="472.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="487.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<line x1="365.0" y1="512.0" x2="367.0" y2="535.0" stroke="#ca6f1e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="322.0" y="570.0" font-size="8.5" fill="#1c2833">H0 有映射，但没有访问权限</text>
<line x1="320.0" y1="592.0" x2="594.0" y2="592.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="320.0" y="608.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="320.0" y="628.0" font-size="8" fill="#566573">VA</text>
<text x="320.0" y="672.0" font-size="8" fill="#566573">HBM</text>
<rect x="352.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="365.0" y="627.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">H0</text>
<rect x="380.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="408.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="436.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="352.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="367.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="392.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="407.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="432.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="447.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="472.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="487.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<line x1="365.0" y1="634.0" x2="367.0" y2="657.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="322.0" y="692.0" font-size="8.5" fill="#1c2833">H0 映射变为可读写；段不变</text>
<text x="322.0" y="722.0" font-size="10" font-weight="bold" fill="#566573">reserved 不变</text>
<rect x="614.0" y="414.0" width="290.0" height="316.0" rx="8" fill="#ffffff" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="624.0" y="432.0" font-size="12" font-weight="bold" font-family="Menlo, Consolas, monospace" fill="#1c2833">cuMemUnmap</text>
<text x="624.0" y="445.0" font-size="8.5" fill="#566573">调用点：VirDevPtr 析构（h:304）</text>
<text x="624.0" y="459.0" font-size="8.5" fill="#33414e">删除 VA 上的映射。HBM 不释放，chunk 仍在。VA 范围随后由 cuMe</text>
<text x="624.0" y="470.0" font-size="8.5" fill="#33414e">mAddressFree 归还。</text>
<text x="622.0" y="486.0" font-size="9" font-weight="bold" fill="#34495e">调用前</text>
<text x="622.0" y="506.0" font-size="8" fill="#566573">VA</text>
<text x="622.0" y="550.0" font-size="8" fill="#566573">HBM</text>
<rect x="654.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="667.0" y="505.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">H0</text>
<rect x="682.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="695.0" y="505.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">H1</text>
<rect x="710.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="738.0" y="492.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="654.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="669.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="694.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="709.0" y="549.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="734.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="749.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="774.0" y="536.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="789.0" y="549.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<line x1="667.0" y1="512.0" x2="669.0" y2="535.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<line x1="695.0" y1="512.0" x2="709.0" y2="535.0" stroke="#34495e" stroke-width="1.2" marker-end="url(#arr)"/>
<text x="624.0" y="570.0" font-size="8.5" fill="#1c2833">H0、H1 有映射</text>
<line x1="622.0" y1="592.0" x2="896.0" y2="592.0" stroke="#d5dbdb" stroke-width="1.2"/>
<text x="622.0" y="608.0" font-size="9" font-weight="bold" fill="#34495e">调用后</text>
<text x="622.0" y="628.0" font-size="8" fill="#566573">VA</text>
<text x="622.0" y="672.0" font-size="8" fill="#566573">HBM</text>
<rect x="654.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="682.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="710.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="738.0" y="614.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<rect x="654.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="669.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<rect x="694.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="709.0" y="671.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H1</text>
<rect x="734.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="749.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H2</text>
<rect x="774.0" y="658.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="789.0" y="671.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H3</text>
<text x="624.0" y="692.0" font-size="8.5" fill="#1c2833">映射删除，VA 还预留；chunk 仍在 HBM；段的 vir_blocks 项移除</text>
<text x="624.0" y="722.0" font-size="10" font-weight="bold" fill="#566573">reserved 不变</text>
<rect x="916.0" y="414.0" width="290.0" height="316.0" rx="8" fill="#fbfcfc" stroke="#bbbbbb" stroke-width="1.2"/>
<text x="926.0" y="432.0" font-size="12" font-weight="bold" fill="#1c2833">图例</text>
<rect x="926.0" y="441.0" width="26.0" height="20.0" rx="2" fill="#d5f5e3" stroke="#1e8449" stroke-width="1.1"/>
<text x="939.0" y="454.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#1e8449">H0</text>
<text x="968.0" y="451.0" font-size="8.5" fill="#1c2833">VA 格子（绿）：已映射，可读写；格子里是 chunk 编号</text>
<rect x="926.0" y="471.0" width="26.0" height="20.0" rx="2" fill="#fdebd0" stroke="#ca6f1e" stroke-width="1.1" stroke-dasharray="3 2"/>
<text x="939.0" y="484.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ca6f1e">H0</text>
<text x="968.0" y="481.0" font-size="8.5" fill="#1c2833">VA 格子（橙虚线）：已映射，没有访问权限</text>
<rect x="926.0" y="501.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#7f8c8d" stroke-width="1"/>
<text x="968.0" y="511.0" font-size="8.5" fill="#1c2833">VA 格子（空心）：预留，没有映射</text>
<rect x="926.0" y="531.0" width="26.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="2 2"/>
<text x="968.0" y="541.0" font-size="8.5" fill="#1c2833">VA 格子（虚线空框）：没有 VA</text>
<rect x="926.0" y="561.0" width="30.0" height="20.0" rx="2" fill="#7d3c98" stroke="#4d4d4d" stroke-width="0.8"/>
<text x="941.0" y="574.0" font-size="8" font-weight="bold" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#ffffff">H0</text>
<text x="968.0" y="571.0" font-size="8.5" fill="#1c2833">HBM 方框（紫）：有物理 chunk（2 MB）</text>
<rect x="926.0" y="591.0" width="30.0" height="20.0" rx="2" fill="#ffffff" stroke="#aab7b8" stroke-width="0.8" stroke-dasharray="3 2"/>
<text x="941.0" y="604.0" font-size="8" text-anchor="middle" font-family="Menlo, Consolas, monospace" fill="#aab7b8">H0</text>
<text x="968.0" y="601.0" font-size="8.5" fill="#1c2833">HBM 方框（虚线空框）：没有 chunk</text>
<text x="926.0" y="640.0" font-size="8.5" fill="#566573">箭头：VA 格子 → 它映射的 HBM chunk。</text>
<text x="926.0" y="664.0" font-size="8.5" fill="#566573">上半部分 = 调用前，下半部分 = 调用后。</text>
<text x="926.0" y="688.0" font-size="8.5" fill="#566573">“段”的变化写在每半右侧的文字里。</text>
<text x="926.0" y="712.0" font-size="8.5" fill="#566573">reserved 只有 cuMemRelease 会下降。</text>
<text x="10.0" y="756.0" font-size="10" fill="#566573">行号来自 GMLake patch 的 cuda_vmm_allocator.h。手推的状态变化，未运行。</text>
</svg>
```
