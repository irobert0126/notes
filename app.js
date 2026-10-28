(function () {
  const BASE = new URL('./', location.href);
  const navEl = document.getElementById('nav');
  const searchEl = document.getElementById('search');
  const noteEl = document.getElementById('note');
  const bodyEl = document.getElementById('body');
  const chipsEl = document.getElementById('chips');
  const footEl = document.getElementById('foot');
  const listing = document.getElementById('listing');
  const dialog = document.getElementById('viewer');
  const big = document.getElementById('big');
  const level = document.getElementById('level');
  const scroller = document.getElementById('scroller');

  let nav = null;
  let notesByPath = new Map();
  let searchIndex = null;
  let searchTimer = null;
  let scale = 1;

  function setTitle(t) {
    document.title = t + ' · notes';
  }

  function hashPath() {
    return decodeURIComponent(location.hash.replace(/^#\/?/, ''));
  }

  function markSelected(path) {
    navEl.querySelectorAll('a[data-path]').forEach((a) => {
      a.classList.toggle('selected', a.dataset.path === path);
    });
  }

  function navLink(path, title) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '#/' + path;
    a.dataset.path = path;
    a.textContent = title;
    li.appendChild(a);
    return li;
  }

  function buildNav() {
    navEl.replaceChildren();
    nav.topics.forEach((t) => {
      const det = document.createElement('details');
      det.open = true;
      const sum = document.createElement('summary');
      sum.textContent = t.title;
      const ul = document.createElement('ul');
      t.notes.forEach((n) => ul.appendChild(navLink(n.path, n.title)));
      det.append(sum, ul);
      navEl.appendChild(det);
    });
    if (nav.checkpoints.length) {
      const det = document.createElement('details');
      det.open = true;
      const sum = document.createElement('summary');
      sum.textContent = 'Checkpoints';
      const ul = document.createElement('ul');
      nav.checkpoints.forEach((c) => ul.appendChild(navLink(c.path, c.title)));
      det.append(sum, ul);
      navEl.appendChild(det);
    }
  }

  function showHome() {
    noteEl.hidden = true;
    listing.hidden = false;
    setTitle('首页');
    markSelected('');
    const h = document.createElement('h1');
    h.textContent = '知识库';
    const p = document.createElement('p');
    p.className = 'lead';
    const total = nav.topics.reduce((s, t) => s + t.notes.length, 0);
    p.textContent = `${nav.topics.length} 个主题，${total} 篇笔记。从左侧选择，或搜索关键词。`;
    const ul = document.createElement('ul');
    ul.className = 'grid-topics';
    nav.topics.forEach((t) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#/' + t.notes[0]?.path;
      if (!t.notes.length) a.removeAttribute('href');
      const name = document.createElement('strong');
      name.textContent = t.title;
      const count = document.createElement('div');
      count.className = 'lead';
      count.textContent = `${t.notes.length} 篇笔记`;
      a.append(name, count);
      li.appendChild(a);
      ul.appendChild(li);
    });
    const graphLink = document.createElement('p');
    graphLink.innerHTML = '<a href="graph.html">打开知识图谱 →</a>';
    listing.replaceChildren(h, p, ul, graphLink);
  }

  function rewriteBody(notePath) {
    const noteUrl = new URL(notePath, BASE);
    bodyEl.querySelectorAll('img').forEach((img) => {
      const raw = img.getAttribute('src');
      if (!raw) return;
      img.src = new URL(raw, noteUrl).href;
      img.addEventListener('click', () => openViewer(img.src, img.alt));
    });
    bodyEl.querySelectorAll('a[href]').forEach((a) => {
      const raw = a.getAttribute('href');
      if (raw.startsWith('#')) return;
      if (/^(https?:|mailto:)/.test(raw)) {
        a.target = '_blank';
        a.rel = 'noopener';
        return;
      }
      const resolved = new URL(raw, noteUrl);
      const base = decodeURIComponent(BASE.pathname);
      const p = decodeURIComponent(resolved.pathname);
      if (p.endsWith('.md') && p.startsWith(base)) {
        a.href = '#/' + p.slice(base.length);
      } else {
        a.href = resolved.href;
        a.target = '_blank';
        a.rel = 'noopener';
      }
    });
  }

  const BI_GAP = 48;

  function setBilingualMode(view, img, mode) {
    view.querySelectorAll('.bi-knob button').forEach((b) => b.classList.toggle('on', b.dataset.mode === mode));
    img.style.width = mode === 'both' ? '100%' : '200%';
    if (mode === 'en') {
      const s = view.clientWidth / img.naturalWidth * 2;
      img.style.transform = `translateX(${-(img.naturalWidth + BI_GAP) / 2 * s}px)`;
    } else {
      img.style.transform = 'none';
    }
    view.dataset.mode = mode;
  }

  function bilingualize() {
    bodyEl.querySelectorAll('img').forEach((img) => {
      if (!/-bilingual\.png$/.test(img.getAttribute('src') || '')) return;
      const wrap = document.createElement('div');
      wrap.className = 'bi';
      const knob = document.createElement('div');
      knob.className = 'bi-knob';
      [['zh', '中文'], ['both', '双语'], ['en', 'English']].forEach(([mode, label]) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.dataset.mode = mode;
        b.textContent = label;
        b.addEventListener('click', () => setBilingualMode(view, img, mode));
        knob.appendChild(b);
      });
      const view = document.createElement('div');
      view.className = 'bi-view';
      img.replaceWith(wrap);
      view.appendChild(img);
      wrap.append(knob, view);
      const apply = () => setBilingualMode(view, img, 'zh');
      if (img.complete && img.naturalWidth) apply();
      else img.addEventListener('load', apply);
      window.addEventListener('resize', () => setBilingualMode(view, img, view.dataset.mode || 'zh'));
    });
  }

  function renderFooter(note) {
    footEl.replaceChildren();
    const sections = [
      ['相关笔记', note.related],
      ['来源', note.sources],
    ];
    sections.forEach(([title, items]) => {
      if (!items.length) return;
      const h = document.createElement('h3');
      h.textContent = title;
      const ul = document.createElement('ul');
      items.forEach((p) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#/' + p;
        const n = notesByPath.get(p);
        a.textContent = n ? n.title : p.split('/').pop();
        li.appendChild(a);
        ul.appendChild(li);
      });
      footEl.append(h, ul);
    });
  }

  async function showNote(path) {
    listing.hidden = true;
    noteEl.hidden = false;
    markSelected(path);
    const note = notesByPath.get(path) || { title: path.split('/').pop(), tags: [], related: [], sources: [] };
    setTitle(note.title);
    chipsEl.replaceChildren(...note.tags.map((t) => {
      const s = document.createElement('span');
      s.className = 'chip';
      s.textContent = t;
      return s;
    }));
    let text;
    try {
      const r = await fetch(new URL(path, BASE));
      if (!r.ok) throw new Error(r.status);
      text = await r.text();
    } catch (e) {
      bodyEl.textContent = '无法加载：' + path;
      footEl.replaceChildren();
      return;
    }
    const body = text.replace(/^---\n[\s\S]*?\n---\n/, '');
    bodyEl.innerHTML = window.marked ? marked.parse(body) : '';
    if (!window.marked) bodyEl.textContent = body;
    rewriteBody(path);
    bilingualize();
    renderFooter(note);
    noteEl.scrollTop = 0;
  }

  function route() {
    const path = hashPath();
    if (!path) return showHome();
    if (path.endsWith('.md')) return showNote(path);
    return showHome();
  }

  function fitScale() {
    return Math.min(scroller.clientWidth / big.naturalWidth, scroller.clientHeight / big.naturalHeight);
  }

  function render() {
    big.style.width = big.naturalWidth * scale + 'px';
    level.textContent = Math.round(scale * 100) + '%';
  }

  function setScale(next) {
    scale = Math.min(8, Math.max(0.1, next));
    render();
  }

  function openViewer(src, alt) {
    big.alt = alt || '';
    big.onload = () => setScale(fitScale());
    big.src = src;
    dialog.showModal();
  }

  document.getElementById('close-btn').addEventListener('click', () => dialog.close());
  document.getElementById('zoom-in-btn').addEventListener('click', () => setScale(scale * 1.25));
  document.getElementById('zoom-out-btn').addEventListener('click', () => setScale(scale / 1.25));
  document.getElementById('zoom-fit-btn').addEventListener('click', () => setScale(fitScale()));
  document.getElementById('print-btn').addEventListener('click', () => window.print());
  scroller.addEventListener('wheel', (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault();
    setScale(scale * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
  }, { passive: false });
  dialog.addEventListener('keydown', (e) => {
    if (e.key === '+' || e.key === '=') setScale(scale * 1.25);
    else if (e.key === '-') setScale(scale / 1.25);
    else if (e.key === '0') setScale(fitScale());
  });

  function runSearch(raw) {
    const query = raw.trim();
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      route();
      return;
    }
    const load = searchIndex || (searchIndex = fetch(new URL('search.json', BASE)).then((r) => r.json()));
    load.then((docs) => {
      if (searchEl.value.trim() !== query) return;
      const hits = [];
      docs.forEach((doc) => {
        const title = doc.title.toLowerCase();
        const text = doc.text.toLowerCase();
        const path = doc.path.toLowerCase();
        if (!terms.every((t) => title.includes(t) || path.includes(t) || text.includes(t))) return;
        const score = terms.reduce((s, t) => s + (title.includes(t) ? 20 : 0) + Math.min(text.split(t).length - 1, 50), 0);
        const idx = terms.map((t) => text.indexOf(t)).find((i) => i >= 0);
        const snip = idx === undefined ? '' : doc.text.slice(Math.max(0, idx - 40), idx + 120).replace(/\s+/g, ' ');
        hits.push({ doc, score, snip });
      });
      hits.sort((a, b) => b.score - a.score);
      const h = document.createElement('h1');
      h.textContent = '搜索：' + query;
      const count = document.createElement('p');
      count.className = 'lead';
      count.textContent = hits.length ? `${hits.length} 条结果` : '没有匹配的结果';
      const ul = document.createElement('ul');
      ul.className = 'results';
      hits.slice(0, 50).forEach((hit) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = '#/' + hit.doc.path;
        a.append(
          Object.assign(document.createElement('span'), { className: 'name', textContent: hit.doc.title }),
          Object.assign(document.createElement('span'), { className: 'path', textContent: hit.doc.path }),
        );
        if (hit.snip) a.append(Object.assign(document.createElement('span'), { className: 'snip', textContent: hit.snip }));
        li.appendChild(a);
        ul.appendChild(li);
      });
      noteEl.hidden = true;
      listing.hidden = false;
      listing.replaceChildren(h, count, ul);
    }).catch(() => {
      listing.hidden = false;
      listing.textContent = '搜索需要通过 http 服务打开本目录。';
    });
  }

  searchEl.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => runSearch(searchEl.value), 150);
  });
  searchEl.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      searchEl.value = '';
      runSearch('');
    }
  });

  window.addEventListener('hashchange', route);

  fetch(new URL('nav.json', BASE))
    .then((r) => r.json())
    .then((data) => {
      nav = data;
      nav.topics.forEach((t) => t.notes.forEach((n) => notesByPath.set(n.path, n)));
      buildNav();
      route();
    })
    .catch(() => {
      listing.hidden = false;
      listing.textContent = '无法加载导航。请用 python3 -m http.server 打开本目录。';
    });
})();
