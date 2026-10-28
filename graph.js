(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const VW = 1200;
  const VH = 800;
  const COLORS = { topic: '#2563eb', note: '#16a34a', concept: '#d97706' };
  const BASE_R = { topic: 14, note: 7, concept: 4.5 };
  const TYPE_LABEL = { topic: '主题', note: '笔记', concept: '概念' };

  const svg = document.getElementById('canvas');
  const world = document.getElementById('world');
  const edgeG = document.getElementById('edges');
  const nodeG = document.getElementById('nodes');
  const labelG = document.getElementById('labels');
  const info = document.getElementById('info');
  const countEl = document.getElementById('count');
  const queryEl = document.getElementById('query');
  const topicSel = document.getElementById('topic-sel');
  const sharedOnly = document.getElementById('shared-only');
  const typeBox = { topic: document.getElementById('show-topic'), note: document.getElementById('show-note'), concept: document.getElementById('show-concept') };

  let nodes = [];
  let links = [];
  let neighbors = new Map();
  let active = [];
  let activeLinks = [];
  let view = { k: 1, x: 0, y: 0 };
  let alpha = 1;
  let running = false;
  let drag = null;
  let pan = null;
  let moved = false;
  let focus = null;
  let autoFit = false;
  let hovered = null;

  function toWorld(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const sx = (clientX - rect.left) * (VW / rect.width);
    const sy = (clientY - rect.top) * (VH / rect.height);
    return { x: (sx - view.x) / view.k, y: (sy - view.y) / view.k };
  }

  function applyView() {
    world.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.k})`);
  }

  function isVisible(n) {
    if (!typeBox[n.type].checked) return false;
    const sel = topicSel.value;
    if (sel) {
      if (n.type === 'topic') return n.id === `topic:${sel}`;
      if (n.type === 'note') return n.topic === sel;
    }
    if (n.type === 'concept' && sharedOnly.checked) return n.noteCount >= 2;
    return true;
  }

  function applyFilters() {
    nodes.forEach((n) => { n.hidden = !isVisible(n); });
    nodes.forEach((n) => {
      if (n.type === 'concept' && !n.hidden) {
        const hasVisibleNote = [...neighbors.get(n)].some((o) => o.type === 'note' && !o.hidden);
        if (!hasVisibleNote) n.hidden = true;
      }
    });
    active = nodes.filter((n) => !n.hidden);
    activeLinks = links.filter((l) => !l.s.hidden && !l.t.hidden);
    nodes.forEach((n) => {
      n.el.style.display = n.hidden ? 'none' : '';
      n.labelEl.style.display = n.hidden ? 'none' : '';
    });
    links.forEach((l) => { l.el.style.display = l.s.hidden || l.t.hidden ? 'none' : ''; });
    const notes = active.filter((n) => n.type === 'note').length;
    countEl.textContent = `显示 ${active.length} / ${nodes.length} 个节点，${activeLinks.length} 条边（笔记 ${notes}）`;
    autoFit = true;
    kick(0.6);
  }

  function build(graph) {
    const index = new Map();
    nodes = graph.nodes.map((n, i) => {
      index.set(n.id, i);
      const a = (i / graph.nodes.length) * Math.PI * 2;
      return { ...n, x: VW / 2 + Math.cos(a) * 350, y: VH / 2 + Math.sin(a) * 280, vx: 0, vy: 0, fx: null, fy: null, hidden: false };
    });
    links = graph.edges
      .map((e) => ({ ...e, s: nodes[index.get(e.source)], t: nodes[index.get(e.target)] }))
      .filter((e) => e.s && e.t);
    neighbors = new Map(nodes.map((n) => [n, new Set()]));
    links.forEach((l) => {
      neighbors.get(l.s).add(l.t);
      neighbors.get(l.t).add(l.s);
    });
    nodes.forEach((n) => {
      n.noteCount = [...neighbors.get(n)].filter((o) => o.type === 'note').length;
      n.r = BASE_R[n.type] + Math.min(neighbors.get(n).size, 8) * 0.4;
      n.labelText = n.type === 'topic' || n.label.length <= 22 ? n.label : n.label.slice(0, 21) + '…';
    });

    const topics = nodes.filter((n) => n.type === 'topic');
    topicSel.replaceChildren(new Option('全部', ''), ...topics.map((t) => new Option(t.label, t.id.replace('topic:', ''))));

    edgeG.replaceChildren();
    nodeG.replaceChildren();
    labelG.replaceChildren();
    links.forEach((l) => {
      const line = document.createElementNS(NS, 'line');
      line.setAttribute('class', `edge ${l.kind}`);
      l.el = line;
      edgeG.appendChild(line);
    });
    nodes.forEach((n) => {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('class', 'node');
      c.setAttribute('fill', COLORS[n.type]);
      c.addEventListener('pointerdown', (e) => startDrag(e, n));
      c.addEventListener('pointerenter', () => hover(n));
      c.addEventListener('pointerleave', () => hover(null));
      n.el = c;
      nodeG.appendChild(c);
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('class', `label${n.type === 'topic' ? ' topic' : ''}${n.type === 'concept' ? ' concept' : ''}`);
      t.textContent = n.labelText;
      n.labelEl = t;
      labelG.appendChild(t);
    });
    applyFilters();
    nodes.forEach((n) => {
      const r = Math.max(n.r, 1);
      n.x = Math.max(60, Math.min(VW - 60, n.x));
      n.y = Math.max(60, Math.min(VH - 60, n.y));
      n.el.setAttribute('r', r);
    });
    alpha = 1;
    kick(1);
  }

  function step() {
    const n = active.length;
    for (let i = 0; i < n; i++) {
      const a = active[i];
      for (let j = i + 1; j < n; j++) {
        const b = active[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.5; }
        const d = Math.sqrt(d2);
        const minD = a.r + b.r + (a.type === 'topic' && b.type === 'topic' ? 110 : 12);
        if (d < minD) {
          const push = (minD - d) * 0.5 * alpha + (minD - d) * 0.15;
          const ux = dx / d;
          const uy = dy / d;
          a.vx -= ux * push;
          a.vy -= uy * push;
          b.vx += ux * push;
          b.vy += uy * push;
        }
        const f = (2800 / Math.max(d2, 400)) * alpha;
        a.vx -= (dx / d) * f;
        a.vy -= (dy / d) * f;
        b.vx += (dx / d) * f;
        b.vy += (dy / d) * f;
      }
    }
    activeLinks.forEach((l) => {
      const dx = l.t.x - l.s.x;
      const dy = l.t.y - l.s.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const target = l.kind === 'contains' ? 120 : l.kind === 'tag' ? 80 : 150;
      const f = (d - target) * 0.035 * alpha;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      l.s.vx += fx;
      l.s.vy += fy;
      l.t.vx -= fx;
      l.t.vy -= fy;
    });
    active.forEach((p) => {
      p.vx += (VW / 2 - p.x) * 0.0015 * alpha;
      p.vy += (VH / 2 - p.y) * 0.0015 * alpha;
      if (p.fx !== null) {
        p.x = p.fx;
        p.y = p.fy;
        p.vx = 0;
        p.vy = 0;
      } else {
        p.vx *= 0.55;
        p.vy *= 0.55;
        p.x += p.vx;
        p.y += p.vy;
      }
    });
    alpha *= 0.985;
  }

  function render() {
    activeLinks.forEach((l) => {
      l.el.setAttribute('x1', l.s.x);
      l.el.setAttribute('y1', l.s.y);
      l.el.setAttribute('x2', l.t.x);
      l.el.setAttribute('y2', l.t.y);
    });
    active.forEach((p) => {
      p.el.setAttribute('cx', p.x);
      p.el.setAttribute('cy', p.y);
      p.labelEl.setAttribute('x', p.x + p.r + 3);
      p.labelEl.setAttribute('y', p.y + 4);
    });
  }

  function loop() {
    step();
    render();
    if (alpha > 0.01 || drag) {
      requestAnimationFrame(loop);
    } else {
      running = false;
      declutter();
      if (autoFit && !drag) {
        autoFit = false;
        fitView();
      }
    }
  }

  function kick(boost) {
    alpha = Math.max(alpha, boost);
    if (!running) {
      running = true;
      requestAnimationFrame(loop);
    }
  }

  function fitView() {
    if (!active.length) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    active.forEach((p) => {
      minX = Math.min(minX, p.x - p.r);
      maxX = Math.max(maxX, p.x + p.r + 80);
      minY = Math.min(minY, p.y - p.r);
      maxY = Math.max(maxY, p.y + p.r);
    });
    const w = Math.max(maxX - minX, 1);
    const h = Math.max(maxY - minY, 1);
    const k = Math.min(VW / w, VH / h, 2) * 0.9;
    view.k = k;
    view.x = (VW - w * k) / 2 - minX * k;
    view.y = (VH - h * k) / 2 - minY * k;
    applyView();
  }

  function applyHighlight(center) {
    const keep = center ? new Set([center, ...neighbors.get(center)]) : null;
    active.forEach((n) => {
      const on = !keep || keep.has(n);
      n.el.classList.toggle('dim', !on);
      n.labelEl.classList.toggle('dim', !on);
    });
    activeLinks.forEach((l) => {
      const on = !center || l.s === center || l.t === center;
      l.el.classList.toggle('dim', !on);
    });
  }

  function priority(n) {
    if (n.type === 'topic') return 3;
    if (n.type === 'note') return 2;
    return 1 + n.noteCount * 0.1;
  }

  function declutter() {
    const forced = new Set();
    if (hovered) forced.add(hovered);
    if (focus) {
      forced.add(focus);
      neighbors.get(focus).forEach((o) => forced.add(o));
    }
    const circles = active.map((n) => [n.x - n.r, n.y - n.r, n.x + n.r, n.y + n.r, n]);
    const placed = [];
    const overlaps = (box, own) => [...circles, ...placed].some((b) => {
      if (b[4] === own) return false;
      return box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1];
    });
    [...active].sort((a, b) => priority(b) - priority(a)).forEach((n) => {
      const fs = n.type === 'topic' ? 14 : 11;
      const w = n.labelText.length * fs * 0.7 + 4;
      const x0 = n.x + n.r + 3;
      const y0 = n.y + 4 - fs;
      const box = [x0, y0, x0 + w, y0 + fs + 3, n];
      const show = forced.has(n) || !overlaps(box, n);
      if (show) placed.push(box);
      n.labelEl.style.display = show ? '' : 'none';
    });
  }

  function hover(node) {
    hovered = node;
    if (node) {
      node.labelEl.style.display = '';
    }
    if (focus) return;
    applyHighlight(node);
    if (!node) declutter();
  }

  function setFocus(node) {
    focus = node;
    if (node) {
      applyHighlight(node);
      showInfo(node);
    } else {
      applyHighlight(null);
      info.hidden = true;
    }
    declutter();
  }

  function showInfo(node) {
    info.hidden = false;
    info.replaceChildren();
    const h = document.createElement('h2');
    h.textContent = node.label;
    const kind = document.createElement('div');
    kind.className = 'kind';
    kind.textContent = TYPE_LABEL[node.type] + (node.type === 'concept' ? `（${node.noteCount} 篇笔记）` : '');
    info.append(h, kind);
    if (node.type === 'note') {
      const p = document.createElement('p');
      p.innerHTML = `<a href="index.html#/${node.id}">打开笔记 →</a>`;
      info.appendChild(p);
    }
    const groups = { topic: [], note: [], concept: [] };
    [...neighbors.get(node)].forEach((o) => groups[o.type].push(o));
    [['note', '相关笔记'], ['concept', '概念'], ['topic', '主题']].forEach(([type, title]) => {
      const items = groups[type].sort((a, b) => a.label.localeCompare(b.label));
      if (!items.length) return;
      const hh = document.createElement('h3');
      hh.textContent = title;
      const ul = document.createElement('ul');
      items.forEach((o) => {
        const li = document.createElement('li');
        if (o.type === 'note') {
          const a = document.createElement('a');
          a.href = '#';
          a.textContent = o.label;
          a.addEventListener('click', (e) => { e.preventDefault(); setFocus(o); });
          li.appendChild(a);
        } else {
          li.textContent = o.label;
        }
        ul.appendChild(li);
      });
      info.append(hh, ul);
    });
  }

  function startDrag(e, node) {
    e.stopPropagation();
    e.preventDefault();
    drag = { node, x: e.clientX, y: e.clientY };
    moved = false;
    const p = toWorld(e.clientX, e.clientY);
    node.fx = p.x;
    node.fy = p.y;
    svg.setPointerCapture(e.pointerId);
    kick(0.3);
  }

  svg.addEventListener('pointerdown', (e) => {
    if (drag) return;
    pan = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    moved = false;
    svg.classList.add('panning');
    svg.setPointerCapture(e.pointerId);
  });

  svg.addEventListener('pointermove', (e) => {
    if (drag) {
      if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 4) moved = true;
      const p = toWorld(e.clientX, e.clientY);
      drag.node.fx = p.x;
      drag.node.fy = p.y;
      kick(0.3);
    } else if (pan) {
      const rect = svg.getBoundingClientRect();
      const dx = (e.clientX - pan.x) * (VW / rect.width);
      const dy = (e.clientY - pan.y) * (VH / rect.height);
      if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
      view.x = pan.vx + dx;
      view.y = pan.vy + dy;
      applyView();
    }
  });

  function endPointer() {
    if (drag) {
      const node = drag.node;
      node.fx = null;
      node.fy = null;
      if (!moved) setFocus(focus === node ? null : node);
      drag = null;
      kick(0.2);
    }
    if (pan) {
      if (!moved && focus) setFocus(null);
      pan = null;
      svg.classList.remove('panning');
    }
  }
  svg.addEventListener('pointerup', endPointer);
  svg.addEventListener('pointercancel', endPointer);

  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (VW / rect.width);
    const sy = (e.clientY - rect.top) * (VH / rect.height);
    const before = { x: (sx - view.x) / view.k, y: (sy - view.y) / view.k };
    view.k = Math.min(5, Math.max(0.25, view.k * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
    view.x = sx - before.x * view.k;
    view.y = sy - before.y * view.k;
    applyView();
  }, { passive: false });

  function search(query) {
    const q = query.trim().toLowerCase();
    nodes.forEach((n) => n.el.classList.remove('match'));
    if (!q) {
      applyHighlight(null);
      return;
    }
    const hits = active.filter((n) => n.label.toLowerCase().includes(q));
    hits.forEach((n) => n.el.classList.add('match'));
    const keep = new Set(hits);
    hits.forEach((n) => neighbors.get(n).forEach((o) => { if (!o.hidden) keep.add(o); }));
    active.forEach((n) => {
      const on = keep.has(n);
      n.el.classList.toggle('dim', !on);
      n.labelEl.classList.toggle('dim', !on);
    });
    activeLinks.forEach((l) => l.el.classList.toggle('dim', !(keep.has(l.s) && keep.has(l.t))));
  }

  queryEl.addEventListener('input', () => search(queryEl.value));
  queryEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const q = queryEl.value.trim().toLowerCase();
    const hit = active.find((n) => n.label.toLowerCase().includes(q));
    if (hit) setFocus(hit);
  });
  queryEl.addEventListener('search', () => { if (!queryEl.value) search(''); });

  Object.values(typeBox).forEach((box) => box.addEventListener('change', applyFilters));
  sharedOnly.addEventListener('change', applyFilters);
  topicSel.addEventListener('change', () => { setFocus(null); applyFilters(); });
  document.getElementById('fit').addEventListener('click', fitView);
  document.getElementById('relayout').addEventListener('click', () => {
    nodes.forEach((n) => {
      n.x = VW / 2 + (Math.random() - 0.5) * 500;
      n.y = VH / 2 + (Math.random() - 0.5) * 400;
      n.vx = 0;
      n.vy = 0;
    });
    alpha = 1;
    autoFit = true;
    kick(1);
  });

  fetch('graph.json')
    .then((r) => r.json())
    .then((g) => {
      build(g);
    })
    .catch(() => {
      info.hidden = false;
      info.textContent = '无法加载 graph.json。请用 python3 -m http.server 打开本目录。';
    });
})();
