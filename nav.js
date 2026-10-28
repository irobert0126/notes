(function () {
  const body = document.body;
  const root = body.dataset.root || '';
  const current = body.dataset.current || '';

  if (window.top !== window) {
    window.parent.postMessage({ navPath: current }, '*');
    return;
  }

  const navEl = document.getElementById('nav');
  const searchEl = document.getElementById('search');
  const frame = document.getElementById('frame');
  const listing = document.getElementById('listing');
  const byPath = new Map();
  const rootPath = decodeURIComponent(new URL(root || './', location.href).pathname);
  let searchIndex = null;
  let searchTimer = null;

  function register(node) {
    byPath.set(node.path, node);
    (node.children || []).forEach(register);
  }

  function span(cls, text) {
    const s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    return s;
  }

  function makeLink(node) {
    const a = document.createElement('a');
    a.dataset.path = node.path;
    a.href = '#' + node.path;
    return a;
  }

  function buildNode(node) {
    const li = document.createElement('li');
    if (node.type !== 'dir') {
      const a = makeLink(node);
      a.textContent = node.title;
      li.appendChild(a);
      return li;
    }
    const details = document.createElement('details');
    details.dataset.path = node.path;
    const summary = document.createElement('summary');
    const a = makeLink(node);
    a.textContent = node.title;
    a.addEventListener('click', (e) => {
      e.preventDefault();
      location.hash = '#' + node.path;
    });
    summary.appendChild(a);
    details.appendChild(summary);
    const ul = document.createElement('ul');
    node.children.forEach((c) => ul.appendChild(buildNode(c)));
    details.appendChild(ul);
    li.appendChild(details);
    return li;
  }

  function markSelected(path) {
    navEl.querySelectorAll('a[data-path]').forEach((a) => {
      a.classList.toggle('selected', a.dataset.path === path);
    });
  }

  function expandTo(path) {
    const a = navEl.querySelector(`a[data-path="${path}"]`);
    let d = a ? a.closest('details') : null;
    while (d) {
      d.open = true;
      d = d.parentElement.closest('details');
    }
  }

  function card(c) {
    const item = document.createElement('li');
    const a = makeLink(c);
    const hint = c.type === 'dir'
      ? `${c.children.length} 项 →`
      : '页面 →';
    a.append(span('name', c.title), span('hint', hint));
    item.appendChild(a);
    return item;
  }

  function renderListing(node) {
    const heading = document.createElement('h1');
    heading.textContent = node.title;
    const grid = document.createElement('ul');
    grid.className = 'grid';
    node.children.forEach((c) => grid.appendChild(card(c)));
    listing.replaceChildren(heading, grid);
  }

  function select(path) {
    const isMd = path.endsWith('.md') && !byPath.has(path);
    const node = isMd
      ? { type: 'md', path, title: path.split('/').pop() }
      : byPath.get(path) || byPath.get(current);
    if (!node) return;
    markSelected(node.path);
    expandTo(node.path);
    document.title = node.title + ' · notes';
    if (node.type === 'dir') {
      frame.hidden = true;
      listing.hidden = false;
      renderListing(node);
      return;
    }
    listing.hidden = true;
    frame.hidden = false;
    if (frame.dataset.path !== node.path) {
      frame.dataset.path = node.path;
      frame.src = node.type === 'md'
        ? root + 'md.html?f=' + encodeURIComponent(node.path)
        : root + node.path;
    }
  }

  frame.addEventListener('load', () => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(frame.contentWindow.location.href).pathname);
    } catch (e) {
      return;
    }
    if (!pathname.startsWith(rootPath)) return;
    const rel = pathname.slice(rootPath.length);
    const node = byPath.get(rel);
    if (!node || node.type !== 'page' || rel === frame.dataset.path) return;
    frame.dataset.path = rel;
    history.replaceState(null, '', '#' + rel);
    markSelected(rel);
    expandTo(rel);
    document.title = node.title + ' · notes';
  });

  window.addEventListener('message', (e) => {
    if (e.source !== frame.contentWindow || !e.data || typeof e.data.navPath !== 'string') return;
    location.hash = '#' + e.data.navPath;
  });

  function fromHash() {
    return decodeURIComponent(location.hash.slice(1));
  }

  window.addEventListener('hashchange', () => select(fromHash() || current));

  function loadSearchIndex() {
    if (!searchIndex) {
      searchIndex = fetch(root + 'search.json').then((r) => r.json());
    }
    return searchIndex;
  }

  function countOf(haystack, term) {
    return haystack.split(term).length - 1;
  }

  function snippetAt(text, idx, len) {
    const start = Math.max(0, idx - 40);
    const end = Math.min(text.length, idx + len + 100);
    return (start > 0 ? '…' : '') + text.slice(start, end).replace(/\s+/g, ' ') + (end < text.length ? '…' : '');
  }

  function renderResults(query, hits) {
    const heading = document.createElement('h1');
    heading.textContent = '搜索：' + query;
    const count = document.createElement('p');
    count.className = 'lead';
    count.textContent = hits.length ? `${hits.length} 条结果` : '没有匹配的结果';
    const ul = document.createElement('ul');
    ul.className = 'results';
    hits.forEach((h) => {
      const li = document.createElement('li');
      const a = makeLink(h.doc);
      a.append(span('name', h.doc.title), span('path', h.doc.path));
      if (h.snip) a.append(span('snip', h.snip));
      li.appendChild(a);
      ul.appendChild(li);
    });
    listing.replaceChildren(heading, count, ul);
    frame.hidden = true;
    listing.hidden = false;
  }

  function runSearch(raw) {
    const query = raw.trim();
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      select(fromHash() || current);
      return;
    }
    loadSearchIndex()
      .then((docs) => {
        if (searchEl.value.trim() !== query) return;
        const hits = [];
        docs.forEach((doc) => {
          const title = doc.title.toLowerCase();
          const text = doc.text.toLowerCase();
          const path = doc.path.toLowerCase();
          if (!terms.every((t) => title.includes(t) || path.includes(t) || text.includes(t))) return;
          const score = terms.reduce((s, t) => s + (title.includes(t) ? 20 : 0) + Math.min(countOf(text, t), 50), 0);
          let snip = '';
          const idx = terms.map((t) => text.indexOf(t)).find((i) => i >= 0);
          if (idx !== undefined) snip = snippetAt(doc.text, idx, 0);
          hits.push({ doc, score, snip });
        });
        hits.sort((a, b) => b.score - a.score);
        renderResults(query, hits.slice(0, 50));
      })
      .catch(() => {
        listing.hidden = false;
        frame.hidden = true;
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

  fetch(root + 'nav.json')
    .then((r) => r.json())
    .then((tree) => {
      register(tree);
      const ul = document.createElement('ul');
      tree.children.forEach((c) => ul.appendChild(buildNode(c)));
      navEl.appendChild(ul);
      select(fromHash() || current);
    })
    .catch(() => {
      listing.hidden = false;
      listing.textContent = '无法加载导航。请用 python3 -m http.server 打开本目录。';
    });
})();
