#!/usr/bin/env python3
"""Build site-nav.json, site-search.json and graph.json from topics/ notes plus graph_overlay.json."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ROOT / "topics"
CHECKPOINTS = ROOT / "checkpoints"
OVERLAY = ROOT / "graph_overlay.json"
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def rel(p):
    return p.relative_to(ROOT).as_posix()


def parse_front(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key.strip()] = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
        else:
            meta[key.strip()] = val.strip("\"'")
    return meta, text[m.end():]


def plain(md):
    t = re.sub(r"```.*?```", " ", md, flags=re.S)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"[#>*|]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_overlay():
    if not OVERLAY.exists():
        return {}
    return json.loads(OVERLAY.read_text(encoding="utf-8"))


def exists_rel(base, target):
    if target.startswith(("http://", "https://", "#", "mailto:")):
        return True
    path = (base / target).resolve() if not target.startswith("/") else ROOT / target.lstrip("/")
    return path.exists()


def load_notes(overlay, errors):
    notes = []
    for md in sorted(TOPICS.rglob("*.md")):
        path = rel(md)
        text = md.read_text(encoding="utf-8")
        meta, body = parse_front(text)
        topic = md.parent.relative_to(TOPICS).as_posix()
        note = {
            "path": path,
            "title": meta.get("title", md.stem),
            "topic": topic,
            "tags": list(meta.get("tags", [])) + overlay.get("add_tags", {}).get(path, []),
            "related": meta.get("related", []),
            "sources": meta.get("sources", []),
            "text": plain(body),
        }
        for r in note["related"]:
            if not (ROOT / r).exists():
                errors.append(f"{path}: related 不存在 {r}")
        for s in note["sources"]:
            if not (ROOT / s).exists():
                errors.append(f"{path}: sources 不存在 {s}")
        for img in IMG_RE.findall(body):
            if not exists_rel(md.parent, img):
                errors.append(f"{path}: 图片不存在 {img}")
        for link in LINK_RE.findall(body):
            if link.endswith(".md") and not exists_rel(md.parent, link):
                errors.append(f"{path}: 链接不存在 {link}")
        notes.append(note)
    return notes


def load_checkpoints():
    items = []
    if not CHECKPOINTS.exists():
        return items
    for md in sorted(CHECKPOINTS.glob("*.md")):
        items.append({"path": rel(md), "title": md.stem, "text": plain(md.read_text(encoding="utf-8"))})
    return items


def build_graph(notes, overlay, errors):
    topic_titles = overlay.get("topic_titles", {})
    concept_labels = overlay.get("concept_labels", {})
    hidden = set(overlay.get("hide_tags", []))
    nodes, edges, seen = [], [], set()

    def add_node(node):
        if node["id"] not in seen:
            seen.add(node["id"])
            nodes.append(node)

    def add_edge(src, dst, kind, label=""):
        key = (kind,) + tuple(sorted([src, dst]))
        if key in seen:
            return
        seen.add(key)
        edges.append({"source": src, "target": dst, "kind": kind, "label": label})

    for topic in sorted({n["topic"] for n in notes}):
        add_node({"id": f"topic:{topic}", "type": "topic", "label": topic_titles.get(topic, topic)})
    for n in notes:
        add_node({"id": n["path"], "type": "note", "label": n["title"], "topic": n["topic"]})
        add_edge(f"topic:{n['topic']}", n["path"], "contains")
        for t in n["tags"]:
            if t in hidden:
                continue
            add_node({"id": f"tag:{t}", "type": "concept", "label": concept_labels.get(t, t)})
            add_edge(n["path"], f"tag:{t}", "tag")
        for r in n["related"]:
            add_edge(n["path"], r, "related")

    for e in overlay.get("extra_edges", []):
        src, dst = e.get("from"), e.get("to")
        if src not in seen or dst not in seen:
            errors.append(f"graph_overlay.json: extra_edges 节点不存在 {src} -> {dst}")
            continue
        add_edge(src, dst, "custom", e.get("label", ""))

    return {"nodes": nodes, "edges": edges}


def main():
    errors = []
    overlay = load_overlay()
    notes = load_notes(overlay, errors)
    checkpoints = load_checkpoints()
    graph = build_graph(notes, overlay, errors)

    by_topic = {}
    for n in notes:
        by_topic.setdefault(n["topic"], []).append(n)
    nav = {
        "topics": [
            {
                "name": t,
                "title": overlay.get("topic_titles", {}).get(t, t),
                "notes": [
                    {"path": n["path"], "title": n["title"], "tags": n["tags"], "related": n["related"], "sources": n["sources"]}
                    for n in sorted(items, key=lambda x: x["title"])
                ],
            }
            for t, items in sorted(by_topic.items())
        ],
        "checkpoints": [{"path": c["path"], "title": c["title"]} for c in checkpoints],
    }
    search = [
        {"path": n["path"], "title": n["title"], "topic": n["topic"], "text": n["text"]} for n in notes
    ] + [{"path": c["path"], "title": c["title"], "topic": "checkpoints", "text": c["text"]} for c in checkpoints]

    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)

    (ROOT / "site-nav.json").write_text(json.dumps(nav, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "site-search.json").write_text(json.dumps(search, ensure_ascii=False), encoding="utf-8")
    (ROOT / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"notes={len(notes)} checkpoints={len(checkpoints)} nodes={len(graph['nodes'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()
