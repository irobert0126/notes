# notes 知识库：精简上下文

## 元信息

- 仓库：`~/notes`，远端 `github.com/irobert0126/notes`（main 分支，GitHub Pages 自动部署）。
- 线上：https://irobert0126.github.io/notes/ （阅读页 `index.html`，图谱 `graph.html`）。
- 用户：Robert Luo，ML 工程师，关注 GPU 架构、FlashAttention、PP、显存管理。
- 原始材料来自 Meta 内部 paste（`knowledge_load` 读取）。

## 数据流

`sources/`（原始 md，只增不改）→ `topics/<主题>/*.md`（整理笔记，带 front-matter）→ `tools/build_site.py` → `nav.json` / `search.json` / `graph.json`。

图谱补充信息在 `graph_overlay.json`，不改笔记文件。

## 更新原则

1. 原始材料只增不改。一个 paste 的嵌套 paste（历史版本、引用的实测数据）也要保存到 `sources/`。
2. 笔记正文只放两部分：一句话目的，和核心知识点（用户原话，已润色）。其余内容留在 `sources/`，笔记里链接。
3. front-matter 必须有 `title`、`topic`、`tags`、`related`、`sources`。`related` 和 `sources` 指向的文件必须存在。
4. 图片放 `images/`。双语图命名为 `<name>-bilingual.png`，左中文右 English，阅读页会自动加“中文 / 双语 / English”切换。单语图保留 `-zh` / `-en` 后缀。
5. 图谱只读 `graph.json`。想加标签、标题或额外边，写在 `graph_overlay.json`。
6. 旧地址不做兼容跳转，已经删除。
7. 不确定的内容标为未验证，不要补充来源里没有的结论。
8. 不把图片传到外部图床。内部分享用 `meta pixelcloud.image upload`。

## 常用操作

- 读取 paste：`knowledge_load`（URL 形如 `https://www.internalfb.com/phabricator/paste/view/P…`）。保存前去掉工具包装行：开头的 `title:`、`doc_type:`、`content:`，结尾的 `creation_time:`、`update_time:`、`language:`、`<!-- devmate-offloaded -->`。
- 新笔记：在 `topics/<主题>/` 下新建 `<slug>.md`，主题文件夹名用英文小写加连字符。
- 构建校验：`python3 tools/build_site.py`。有缺失引用会报错并退出，不写出任何文件。
- 图片渲染 SVG：用无头 Chrome，`"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-sandbox --disable-gpu --window-size=W,H --screenshot=out.png file:///path/wrapper.html`，wrapper 里用 `<img width=2W>` 放大 2 倍。
- JS 语法检查：`osascript -l JavaScript`，用 `new Function(src)` 解析。
- 页面验证：`python3 -m http.server 8000`，再用无头 Chrome `--dump-dom` 检查渲染结果。
- 提交：`git add -A && git commit -m … && git pull --rebase && git push`。

## 当前状态

- 笔记 12 篇：gpu-architecture 4 篇，attention-kernels 5 篇，pipeline-parallel 2 篇，gpu-memory 1 篇（VMM 与 expandable_segments，含 VMS 小节）。另有 1 篇 checkpoint。
- 图谱默认只显示被多篇笔记共享的概念。
- 已知限制：图谱标签去重是近似计算；页面交互只在无头 Chrome 里验证过基本渲染和点击。
