(function () {
  const body = document.body;
  const root = body.dataset.root || '';
  const current = body.dataset.current || '';

  if (window.top !== window) {
    window.parent.postMessage({ navPath: current }, '*');
    return;
  }

  const navEl = document.getElementById('nav');
  const frame = document.getElementById('frame');
  const listing = document.getElementById('listing');
  const byPath = new Map();
  const rootPath = decodeURIComponent(new URL(root || './', location.href).pathname);

  function register(node) {
    byPath.set(node.path, node);
    (node.children || []).forEach(register);
  }

  function makeLink(node) {
    const a = document.createElement('a');
    a.dataset.path = node.path;
    if (node.type === 'md') {
      a.href = root + node.path;
      a.target = '_blank';
      a.rel = 'noopener';
    } else {
      a.href = '#' + node.path;
    }
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
    const parts = path ? path.split('/') : [];
    for (let i = 1; i <= parts.length; i++) {
      const details = navEl.querySelector(`details[data-path="${parts.slice(0, i).join('/')}"]`);
      if (details) details.open = true;
    }
  }

  function renderListing(node) {
    const heading = document.createElement('h1');
    heading.textContent = node.title;
    const grid = document.createElement('ul');
    grid.className = 'grid';
    node.children.forEach((c) => {
      const item = document.createElement('li');
      const a = makeLink(c);
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = c.title;
      const hint = document.createElement('span');
      hint.className = 'hint';
      hint.textContent = c.type === 'dir' ? `${c.children.length} 项 →` : c.type === 'md' ? 'Markdown ↗' : '页面 →';
      a.append(name, hint);
      item.appendChild(a);
      grid.appendChild(item);
    });
    listing.replaceChildren(heading, grid);
  }

  function select(path) {
    const node = byPath.get(path) || byPath.get(current);
    if (!node) return;
    if (node.type === 'md') {
      window.open(root + node.path, '_blank');
      return;
    }
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
      frame.src = root + node.path;
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
