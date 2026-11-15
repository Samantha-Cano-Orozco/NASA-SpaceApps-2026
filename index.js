(function () {
  'use strict';

  var D = window.DATA;
  var MARS_ZOOM = 3;
  var marsZone = D.mars.zone;
  var selection = { mars: null, earth: null };

  // ---------- Starfield ----------
  var starCanvas = document.getElementById('stars');
  var sctx = starCanvas.getContext('2d');
  var stars = [];

  function seedStars() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    starCanvas.width = Math.round(window.innerWidth * dpr);
    starCanvas.height = Math.round(window.innerHeight * dpr);
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    stars = [];
    var count = Math.round((window.innerWidth * window.innerHeight) / 1800);
    for (var i = 0; i < count; i++) {
      var r = Math.random();
      stars.push({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        r: r < 0.92 ? Math.random() * 0.8 + 0.2 : Math.random() * 1.2 + 0.9,
        a: Math.random() * 0.6 + 0.25,
        tw: Math.random() * Math.PI * 2,
        s: Math.random() * 0.02 + 0.004
      });
    }
    drawStars(0);
  }

  function drawStars(t) {
    sctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    for (var i = 0; i < stars.length; i++) {
      var s = stars[i];
      var a = s.a * (0.75 + 0.25 * Math.sin(s.tw + t * s.s));
      sctx.beginPath();
      sctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      sctx.fillStyle = 'rgba(214,226,242,' + a.toFixed(3) + ')';
      sctx.fill();
    }
  }

  var animT = 0;
  function loop() {
    animT += 16;
    drawStars(animT);
    requestAnimationFrame(loop);
  }
  seedStars();
  window.addEventListener('resize', seedStars);
  if (!window.matchMedia || !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    requestAnimationFrame(loop);
  }

  // ---------- Globes ----------
  var marsGlobe = createGlobe({
    canvas: document.getElementById('marsCanvas'),
    texture: window.TEXTURES.mars,
    lat: marsZone.lat, lon: marsZone.lon, zoom: 1,
    zones: [{ id: marsZone.id, lat: marsZone.lat, lon: marsZone.lon, radius: marsZone.radius, color: '#d9733f' }],
    onSelect: function (item) { choose('mars', item.id); },
    onHover: function (item) { hover('mars', item); }
  });

  var earthGlobe = createGlobe({
    canvas: document.getElementById('earthCanvas'),
    texture: window.TEXTURES.earth,
    lat: 20, lon: -95, zoom: 1,
    markers: D.sites.map(function (s) {
      return { id: s.id, lat: s.lat, lon: s.lon, color: '#4aa3df' };
    }),
    onSelect: function (item) { choose('earth', item.id); },
    onHover: function (item) { hover('earth', item); }
  });
  earthGlobe.setEnabled(false);

  var globes = { mars: marsGlobe, earth: earthGlobe };

  // ---------- Lookups ----------
  function siteById(id) {
    return D.sites.find(function (s) { return s.id === id; }) || null;
  }

  // ---------- Hover readout (side panel) ----------
  var readout = document.getElementById('readout');

  function hover(kind, item) {
    if (!item) {
      showIdle();
      return;
    }
    if (kind === 'mars') {
      showText('mars', marsZone.name, 'Meridiani Planum · Mars', null);
      return;
    }
    if (kind === 'earth') {
      var s = siteById(item.id);
      if (s) showText('earth', s.name, s.state + ', ' + s.country, 'imagenes/' + s.img + '.jpg');
    }
  }

  function showText(kind, title, sub, img) {
    readout.className = 'readout-body is-live';
    readout.dataset.kind = kind;
    readout.innerHTML = '<div class="r-media"><img alt=""></div><div class="r-title"></div><div class="r-sub"></div>';
    readout.querySelector('.r-title').textContent = title;
    readout.querySelector('.r-sub').textContent = sub;
    var media = readout.querySelector('.r-media');
    var image = readout.querySelector('img');
    if (!img) { media.remove(); return; }   // Mars: no image
    image.alt = title;
    image.onload = function () { media.classList.add('ready'); };
    image.onerror = function () { media.remove(); };   // no photo yet: hide the box
    image.src = img;
  }

  function showIdle() {
    readout.className = 'readout-body is-idle';
    readout.removeAttribute('data-kind');
    readout.innerHTML = '<div class="r-title">No target</div><div class="r-sub">Move the cursor over a zone or site</div>';
  }

  // ---------- Selection flow: Mars first, then Earth ----------
  function choose(kind, id) {
    if (kind === 'earth' && !selection.mars) return;
    selection[kind] = id;
    globes[kind].setActive(id);

    if (kind === 'mars') {
      marsGlobe.focus(marsZone.lat, marsZone.lon, MARS_ZOOM, 800);
      earthGlobe.setEnabled(true);
      // After a short pause, scroll down to the Earth section.
      setTimeout(function () {
        document.getElementById('earthFigure').scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 1000);
    } else {
      var s = siteById(id);
      earthGlobe.focus(s.lat, s.lon, 1.8, 800);
    }
    render();
  }

  function render() {
    var marsEl = document.getElementById('marsSel');
    var earthEl = document.getElementById('earthSel');
    var s = selection.earth ? siteById(selection.earth) : null;

    marsEl.textContent = selection.mars ? marsZone.name : 'Not selected';
    marsEl.classList.toggle('empty', !selection.mars);
    earthEl.textContent = s ? s.name : 'Not selected';
    earthEl.classList.toggle('empty', !s);

    document.getElementById('stepMars').classList.toggle('done', !!selection.mars);
    document.getElementById('stepEarth').classList.toggle('done', !!s);
    document.getElementById('stepEarth').classList.toggle('locked', !selection.mars);
    document.getElementById('earthFigure').classList.toggle('locked', !selection.mars);

    var btn = document.getElementById('compareBtn');
    var ready = !!(selection.mars && selection.earth);
    btn.setAttribute('aria-disabled', ready ? 'false' : 'true');
    btn.href = ready
      ? 'compare.html?mars=' + encodeURIComponent(selection.mars) + '&earth=' + encodeURIComponent(selection.earth)
      : '#';
  }

  document.querySelectorAll('[data-zoom]').forEach(function (b) {
    b.addEventListener('click', function () {
      var g = globes[b.dataset.zoom];
      if (b.dataset.dir === 'in') g.zoomIn(); else g.zoomOut();
    });
  });

  showIdle();
  render();
})();
