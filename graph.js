(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const VW = 1200;
  const VH = 800;
  const COLORS = { topic: '#2563eb', note: '#16a34a', concept: '#d97706' };
  const RADIUS = { topic: 13, note: 8, concept: 5 };
  const svg = document.getElementById('canvas');
  const world = document.getElementById('world');
  const edgeG = document.getElementById('edges');
  const nodeG = document.getElementById('nodes');
  const labelG = document.getElementById('labels');
  const info = document.getElementById('info');

  let nodes = [];
  let links = [];
  let neighbors = new Map();
  let view = { k: 1, x: 0, y: 0 };
  let alpha = 1;
  let running = false;
  let drag = null;
  let pan = null;
  let moved = false;

  function toWorld(clientX, clientY) {
    const rect = svg.getBoundingClientRect();
    const sx = (clientX - rect.left) * (VW / rect.width);
    const sy = (clientY - rect.top) * (VH / rect.height);
    return { x: (sx - view.x) / view.k, y: (sy - view.y) / view.k };
  }

  function applyView() {
    world.setAttribute('transform', `translate(${view.x} ${view.y}) scale(${view.k})`);
  }

  function build(graph) {
    const index = new Map();
    nodes = graph.nodes.map((n, i) => {
      index.set(n.id, i);
      const angle = (i / graph.nodes.length) * Math.PI * 2;
      return {
        ...n,
        x: VW / 2 + Math.cos(angle) * 320,
        y: VH / 2 + Math.sin(angle) * 260,
        vx: 0,
        vy: 0,
        fx: null,
        fy: null,
        r: RADIUS[n.type] || 6,
      };
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
      n.r += Math.min(neighbors.get(n).size, 6) * 0.6;
    });

    edgeG.replaceChildren();
    nodeG.replaceChildren();
    labelG.replaceChildren();
    links.forEach((l) => {
      const line = document.createElementNS(NS, 'line');
      line.setAttribute('class', `edge ${l.kind}`);
      line.setAttribute('stroke-width', l.kind === 'tag' ? 0.8 : 1.4);
      l.el = line;
      edgeG.appendChild(line);
    });
    nodes.forEach((n) => {
      const c = document.createElementNS(NS, 'circle');
      c.setAttribute('class', 'node');
      c.setAttribute('fill', COLORS[n.type] || '#888');
      c.addEventListener('pointerdown', (e) => startDrag(e, n));
      c.addEventListener('pointerenter', () => highlight(n));
      c.addEventListener('pointerleave', () => highlight(null));
      n.el = c;
      nodeG.appendChild(c);
      const t = document.createElementNS(NS, 'text');
      t.setAttribute('class', `label${n.type === 'topic' ? ' topic' : ''}`);
      t.textContent = n.label;
      n.labelEl = t;
      labelG.appendChild(t);
    });
    alpha = 1;
    kick();
  }

  function tick() {
    const n = nodes.length;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = nodes[i];
        const b = nodes[j];
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        const d2 = Math.max(dx * dx + dy * dy, 25);
        const f = (2600 / d2) * alpha;
        const d = Math.sqrt(d2);
        dx = (dx / d) * f;
        dy = (dy / d) * f;
        a.vx -= dx;
        a.vy -= dy;
        b.vx += dx;
        b.vy += dy;
      }
    }
    links.forEach((l) => {
      const dx = l.t.x - l.s.x;
      const dy = l.t.y - l.s.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const target = l.kind === 'contains' ? 110 : l.kind === 'tag' ? 90 : 130;
      const f = (d - target) * 0.04 * alpha;
      const fx = (dx / d) * f;
      const fy = (dy / d) * f;
      l.s.vx += fx;
      l.s.vy += fy;
      l.t.vx -= fx;
      l.t.vy -= fy;
    });
    nodes.forEach((p) => {
      p.vx += (VW / 2 - p.x) * 0.002 * alpha;
      p.vy += (VH / 2 - p.y) * 0.002 * alpha;
      if (p.fx !== null) {
        p.x = p.fx;
        p.y = p.fy;
        p.vx = 0;
        p.vy = 0;
      } else {
        p.vx *= 0.6;
        p.vy *= 0.6;
        p.x += p.vx;
        p.y += p.vy;
      }
    });
    alpha *= 0.985;
  }

  function render() {
    links.forEach((l) => {
      l.el.setAttribute('x1', l.s.x);
      l.el.setAttribute('y1', l.s.y);
      l.el.setAttribute('x2', l.t.x);
      l.el.setAttribute('y2', l.t.y);
    });
    nodes.forEach((p) => {
      p.el.setAttribute('cx', p.x);
      p.el.setAttribute('cy', p.y);
      p.el.setAttribute('r', p.r);
      p.labelEl.setAttribute('x', p.x + p.r + 3);
      p.labelEl.setAttribute('y', p.y + 4);
    });
  }

  function loop() {
    tick();
    render();
    if (alpha > 0.004 || drag) {
      requestAnimationFrame(loop);
    } else {
      running = false;
    }
  }

  function kick(boost = 0) {
    alpha = Math.max(alpha, boost);
    if (!running) {
      running = true;
      requestAnimationFrame(loop);
    }
  }

  function highlight(node) {
    nodes.forEach((n) => {
      const on = !node || n === node || neighbors.get(node).has(n);
      n.el.classList.toggle('dim', !on);
      n.labelEl.classList.toggle('dim', !on);
    });
    links.forEach((l) => {
      const on = !node || l.s === node || l.t === node;
      l.el.classList.toggle('dim', !on);
    });
  }

  function showInfo(node) {
    if (!node) {
      info.hidden = true;
      return;
    }
    info.hidden = false;
    info.replaceChildren();
    const h = document.createElement('h2');
    h.textContent = node.label;
    const kind = document.createElement('div');
    kind.className = 'kind';
    kind.textContent = { topic: '主题', note: '笔记', concept: '概念' }[node.type];
    info.append(h, kind);
    const others = [...neighbors.get(node)].sort((a, b) => a.label.localeCompare(b.label));
    if (others.length) {
      const ul = document.createElement('ul');
      others.forEach((o) => {
        const li = document.createElement('li');
        if (o.type === 'note') {
          const a = document.createElement('a');
          a.href = 'index.html#/' + o.id;
          a.textContent = o.label;
          li.appendChild(a);
        } else {
          li.textContent = o.label;
        }
        ul.appendChild(li);
      });
      info.appendChild(ul);
    }
    if (node.type === 'note') {
      const a = document.createElement('p');
      a.innerHTML = `<a href="index.html#/${node.id}">打开笔记 →</a>`;
      info.appendChild(a);
    }
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

  function endPointer(e) {
    if (drag) {
      const node = drag.node;
      node.fx = null;
      node.fy = null;
      if (!moved) {
        showInfo(node);
        if (node.type === 'note') {
          window.location.href = 'index.html#/' + node.id;
        }
      }
      drag = null;
      kick(0.2);
    }
    if (pan) {
      if (!moved) showInfo(null);
      pan = null;
      svg.classList.remove('panning');
    }
  }
  svg.addEventListener('pointerup', endPointer);
  svg.addEventListener('pointercancel', endPointer);

  svg.addEventListener('wheel', (e) => {
    e.preventDefault();
    const before = toWorld(e.clientX, e.clientY);
    view.k = Math.min(4, Math.max(0.3, view.k * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
    const rect = svg.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (VW / rect.width);
    const sy = (e.clientY - rect.top) * (VH / rect.height);
    view.x = sx - before.x * view.k;
    view.y = sy - before.y * view.k;
    applyView();
  }, { passive: false });

  document.getElementById('reset').addEventListener('click', () => {
    view = { k: 1, x: 0, y: 0 };
    applyView();
    build(graphData);
  });

  let graphData = null;
  fetch('graph.json')
    .then((r) => r.json())
    .then((g) => {
      graphData = g;
      applyView();
      build(g);
    })
    .catch(() => {
      info.hidden = false;
      info.textContent = '无法加载 graph.json。请用 python3 -m http.server 打开本目录。';
    });
})();
