#!/usr/bin/env python3
"""Regenerate nav.json, search.json and the shell index.html for every folder in the notes tree."""
import json
import re
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"tools", "images", "__pycache__"}
SKIP_FILES = {"index.html", "README.md", "md.html"}
SEARCH = []

SHELL = """<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="stylesheet" href="{root}style.css">
</head>
<body class="shell" data-root="{root}" data-current="{cur}">
  <aside id="tree">
    <div class="brand"><a href="{root}index.html">notes</a></div>
    <input id="search" type="search" placeholder="搜索标题或内容" autocomplete="off">
    <div id="nav"></div>
  </aside>
  <section id="pane">
    <iframe id="frame" title="内容" hidden></iframe>
    <div id="listing" hidden></div>
  </section>
  <script src="{root}nav.js"></script>
</body>
</html>
"""


def rel(path):
    return "" if path == ROOT else path.relative_to(ROOT).as_posix()


def html_title(path):
    m = re.search(r"<h1>(.*?)</h1>", path.read_text(encoding="utf-8"), re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else path.stem


def html_text(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def build(folder):
    children = []
    md = []
    for p in sorted(folder.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_dir():
            if p.name in SKIP_DIRS:
                continue
            sub = build(p)
            if sub["children"] or sub["md"]:
                children.append(sub)
        elif p.name in SKIP_FILES:
            continue
        elif p.suffix == ".html":
            title = html_title(p)
            children.append({"type": "page", "path": rel(p), "title": title})
            SEARCH.append({"path": rel(p), "title": title, "text": html_text(p)})
        elif p.suffix == ".md":
            md.append({"type": "md", "path": rel(p), "title": p.name})
            SEARCH.append({"path": rel(p), "title": p.name, "text": p.read_text(encoding="utf-8")})
    children.sort(key=lambda c: (c["type"] != "dir", c["title"]))
    md.sort(key=lambda c: c["title"])
    title = "notes" if folder == ROOT else folder.name
    SEARCH.append({"path": rel(folder), "title": title, "text": ""})
    return {"type": "dir", "path": rel(folder), "title": title, "children": children, "md": md}


def write_shells(node):
    cur = node["path"]
    depth = cur.count("/") + 1 if cur else 0
    title = "notes" if not cur else "notes / " + " / ".join(cur.split("/"))
    (ROOT / cur / "index.html").write_text(
        SHELL.format(title=title, root="../" * depth, cur=cur), encoding="utf-8"
    )
    for c in node["children"]:
        if c["type"] == "dir":
            write_shells(c)


def main():
    tree = build(ROOT)
    (ROOT / "nav.json").write_text(json.dumps(tree, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "search.json").write_text(json.dumps(SEARCH, ensure_ascii=False), encoding="utf-8")
    write_shells(tree)


if __name__ == "__main__":
    main()
