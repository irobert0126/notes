# notes

个人知识库，按主题组织。原始材料只增不改，笔记从原始材料整理而来。

## 目录

- `topics/<主题>/*.md`：整理后的笔记，带 front-matter（title、topic、tags、related、sources）。
- `sources/`：原始材料（paste 的 md 等），只增不改，笔记里链接到这里。
- `images/`：笔记引用的图片。
- `checkpoints/`：学习到一半的进度快照。
- `graph_overlay.json`：图谱的补充信息（主题名、概念名、额外标签、额外边）。不需要改笔记文件。
- `tools/build_site.py`：生成 `nav.json`、`search.json`、`graph.json`，并校验链接和图片。
- `index.html`、`app.js`：阅读页（主题树、搜索、Markdown 渲染、图片放大、打印）。
- `graph.html`、`graph.js`：知识图谱（可拖动的力导向图）。

## 更新流程

1. 新材料：保存到 `sources/`，文件名用小写加连字符，例如 `sources/vmm-expandable-segments-v5.md`。图片放到 `images/`。
2. 笔记：在 `topics/<主题>/` 下新建或修改 `.md`。front-matter 里写 `related` 和 `sources`。
3. 图谱补充：只需要改 `graph_overlay.json`，例如给已有笔记加标签、添加主题名、添加额外的边。
4. 运行 `python3 tools/build_site.py`。它会检查所有引用是否存在，有缺失会报错并退出。
5. 本地预览：`python3 -m http.server 8000`，然后打开 `http://localhost:8000/`。
6. 提交并推送，GitHub Pages 会自动部署。

## 约定

- 一个 paste 的嵌套 paste 也要保存到 `sources/`，笔记里列出链接。
- 笔记正文只放整理后的要点；完整的 SVG、脚本和原始数据留在 `sources/`。
- 历史版本（例如 v1–v5）都保留在 `sources/`，笔记引用当前版本，并列出历史版本的链接。
