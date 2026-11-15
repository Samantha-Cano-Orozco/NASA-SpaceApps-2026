(function () {
  'use strict';

  var D = window.DATA;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(window.location.search);

  var marsZone = D.mars.zone;
  var site = D.sites.find(function (s) { return s.id === params.get('earth'); });
  if (!site || params.get('mars') !== marsZone.id) {
    window.location.replace('index.html');
    return;
  }

  var cond = site.conditions;
  var T = site.terrain;
  var bars = [];          // bar fills to animate on load: { el, pct }

  function f(v, d) { return v.toFixed(d === undefined ? 1 : d); }
  function n(v) { return Math.round(v).toLocaleString('en-US'); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function row(cells) {
    var tr = el('tr');
    cells.forEach(function (c) {
      var td = el(c.h ? 'th' : 'td', c.cls, c.text);
      if (c.node) td.appendChild(c.node);
      tr.appendChild(td);
    });
    return tr;
  }
  function tag(text) {
    return el('span', 'tag ' + text.toLowerCase().split(' ')[0], text);
  }
  // Bar that grows to pct (0-100) when the page loads.
  function bar(pct, cls) {
    var track = el('div', 'bar-track');
    var fill = el('div', 'bar-fill' + (cls ? ' ' + cls : ''));
    track.appendChild(fill);
    bars.push({ el: fill, pct: Math.max(0, Math.min(100, pct)) });
    return track;
  }
  // Value followed by its bar, for table cells.
  function pctCell(pct, cls) {
    var wrap = el('div', 'pct-cell');
    wrap.appendChild(el('span', 'pv ' + (cls || ''), f(pct) + '%'));
    wrap.appendChild(bar(pct, cls));
    return wrap;
  }
  function byLabel(label) {
    return T.metrics.filter(function (m) { return m.label === label; })[0];
  }
  // Number that counts up from 0 on load.
  function countUp(node, to, decimals, suffix) {
    var start = null, dur = 1200;
    function step(ts) {
      if (start === null) start = ts;
      var t = Math.min(1, (ts - start) / dur);
      var e = 1 - Math.pow(1 - t, 3);
      node.textContent = (to * e).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // ---------- Title ----------
  document.title = site.name + ' vs ' + marsZone.name + ' · Comparison';
  $('title').textContent = site.name + ' against ' + marsZone.name;

  // ---------- Head to head: the two worlds ----------
  $('earthName').textContent = site.name;
  $('earthPlace').textContent = site.state + ', ' + site.country;
  $('marsName').textContent = marsZone.name;
  $('marsPlace').textContent = D.mars.site + ' · rover ' + D.mars.rover;
  $('simSub').textContent = 'Terrain similarity ' + T.similarity.toFixed(3) + ' · rank ' + T.rank;
  countUp($('simBig'), site.similarity, 1, '%');

  // Main bar: full width = 100%. Filled to the overall similarity, the average of the mineral and terrain similarity.
  var overall = (site.similarity + T.similarity * 100) / 2;
  $('mbValue').textContent = f(overall) + '%';
  $('mbSub').textContent = 'Mineral ' + f(site.similarity) + '% · terrain ' + f(T.similarity * 100) + '% · overall is the average of the two';
  bars.push({ el: $('mbFill'), pct: overall });

  createGlobe({
    canvas: $('earthGlobe'), texture: window.TEXTURES.earth,
    lat: site.lat, lon: site.lon, zoom: 1.6, interactive: false, labels: false,
    markers: [{ id: site.id, lat: site.lat, lon: site.lon, color: '#4aa3df' }]
  }).setActive(site.id);
  createGlobe({
    canvas: $('marsGlobe'), texture: window.TEXTURES.mars,
    lat: marsZone.lat, lon: marsZone.lon, zoom: 1, interactive: false, labels: false,
    zones: [{ id: marsZone.id, lat: marsZone.lat, lon: marsZone.lon, radius: marsZone.radius, color: '#d9733f' }]
  }).setActive(marsZone.id);

  // Statistics between the two worlds. Each row has a bar per side, scaled to the larger value.
  var pairs = [
    { label: 'Median slope', fmt: function (v) { return f(v, 2) + '°'; } },
    { label: '95th percentile slope', fmt: function (v) { return f(v, 2) + '°'; } },
    { label: 'Median roughness (3×3)', fmt: function (v) { return f(v, 2) + ' m'; } },
    { label: 'Terrain with slope below 5°', fmt: function (v) { return f(v, 2) + '%'; } },
    { label: 'Median relief per 1 km cell', fmt: function (v) { return f(v, 1) + ' m'; } }
  ];
  var rows = $('versusRows');
  pairs.forEach(function (p) {
    var m = byLabel(p.label);
    if (!m) return;
    var max = Math.max(m.site, m.victoria) || 1;
    var item = el('div', 'vitem');
    var r = el('div', 'vrow');
    r.appendChild(el('span', 'v earth', p.fmt(m.site)));
    r.appendChild(el('span', 'l', p.label));
    r.appendChild(el('span', 'v mars', p.fmt(m.victoria)));
    var b = el('div', 'vbars');
    var left = el('div', 'half left'), right = el('div', 'half right');
    var lf = el('div', 'bar-fill'), rf = el('div', 'bar-fill mars');
    left.appendChild(lf); right.appendChild(rf);
    bars.push({ el: lf, pct: m.site / max * 100 });
    bars.push({ el: rf, pct: m.victoria / max * 100 });
    b.appendChild(left); b.appendChild(right);
    item.appendChild(r); item.appendChild(b);
    rows.appendChild(item);
  });

  // ---------- Criteria ----------
  var tbody = $('criteriaRows');
  site.criteria.forEach(function (c) {
    var tgt = el('td');
    tgt.appendChild(tag(c.target));
    tbody.appendChild(row([
      { h: true, text: c.criterion },
      { node: tgt },
      { cls: 'muted', text: c.earth },
      { cls: 'n', text: String(c.weight) },
      { cls: 'n', node: pctCell(c.match) },
      { cls: 'n', text: f(c.lost, 2) }
    ]));
  });
  var W = site.criteria.reduce(function (s, c) { return s + c.weight; }, 0);
  var L = site.criteria.reduce(function (s, c) { return s + c.lost; }, 0);
  var sum = row([
    { h: true, text: 'Similarity' }, { text: '' }, { cls: 'muted', text: 'Weighted mean' },
    { cls: 'n', text: String(W) }, { cls: 'n', node: pctCell(site.similarity) }, { cls: 'n', text: f(L, 2) }
  ]);
  sum.className = 'sum';
  tbody.appendChild(sum);

  // ---------- Composition ----------
  $('compNote').textContent = 'Mars column: Meridiani Planum reference';
  $('compEarth').textContent = site.name;
  site.composition.forEach(function (r) {
    var v = r.earth, text;
    if (typeof v === 'number') {
      if (r.unit === '%') text = f(v, v < 1 ? 3 : 2) + '%';
      else if (r.unit === 'm') text = n(v) + ' m';
      else text = v.toFixed(4);
    } else {
      text = v;
    }
    $('compRows').appendChild(row([
      { h: true, text: r.label },
      { cls: 'muted', text: r.mars },
      { cls: 'n', text: text }
    ]));
  });

  // ---------- Surface conditions ----------
  function kv(list, k, v) { list.appendChild(row([{ h: true, text: k }, { text: v }])); }
  var cr = $('condRows');
  kv(cr, 'EMIT pixels with a detection', f(cond.emit_detection_pct) + '%');
  kv(cr, 'Surface water (region)', f(cond.water_pct) + '%');
  kv(cr, 'Water limit per cell', f(cond.max_water_cell_pct) + '%');
  kv(cr, 'Mean elevation', n(cond.elevation_m) + ' m');
  kv(cr, 'Mean slope', f(cond.slope_deg) + '°');
  kv(cr, 'Soil moisture, 8 Nov 2026 (morning)', cond.moisture == null ? '—' : f(cond.moisture, 3) + ' m³/m³');
  kv(cr, 'Quartz threshold', cond.quartz_rule || '—');
  kv(cr, 'Basalt threshold', cond.mafic_rule || '—');
  kv(cr, 'Automatic warnings', cond.warnings || 'none');

  // ---------- EMIT groups ----------
  site.emit.groups.forEach(function (g) {
    $('emitRows').appendChild(row([
      { h: true, text: g.group },
      { cls: 'n', node: pctCell(g.share) },
      { cls: 'n', text: String(g.minerals) },
      { cls: 'n', text: String(g.flagged) }
    ]));
  });
  var sumE = row([
    { h: true, text: 'Total' }, { cls: 'n', text: '100%' },
    { cls: 'n', text: String(site.emit.minerals) }, { cls: 'n', text: String(site.emit.flagged) }
  ]);
  sumE.className = 'sum';
  $('emitRows').appendChild(sumE);

  // ---------- Terrain table ----------
  $('tSiteHead').textContent = site.name;
  T.metrics.forEach(function (m) {
    var d = m.diff;
    $('terrainRows').appendChild(row([
      { h: true, text: m.label + ' (' + m.unit + ')' },
      { cls: 'n', text: m.victoria.toFixed(2) },
      { cls: 'n', text: m.site.toFixed(2) },
      { cls: 'n', text: (d >= 0 ? '+' : '') + d.toFixed(2) },
      { cls: 'n', text: m.rel == null ? '—' : f(m.rel * 100, 1) + '%' }
    ]));
  });
  $('tLine').textContent = 'Terrain similarity ' + T.similarity.toFixed(3) + ' · rank ' + T.rank + ' among regions and zones (1 = closest to Victoria).';
  var tn = $('terrainNotes');
  tn.appendChild(el('li', null, 'Victoria values use the flat plain outside the crater (30 m DEM).'));
  tn.appendChild(el('li', null, 'The site includes ridges and slopes. A flat-plain subset is not available at pixel level for this site.'));
  tn.appendChild(el('li', null, 'Site relief covers the whole analysed area and is not comparable with the Victoria area.'));

  // ---------- Similarity summary (with bars) ----------
  var sr = $('simRows');
  sr.appendChild(row([{ h: true, text: 'Full-region similarity' }, { cls: 'n', node: pctCell(site.similarity) }]));
  sr.appendChild(row([{ h: true, text: 'Median cell similarity' }, { cls: 'n', node: pctCell(site.median) }]));
  sr.appendChild(row([{ h: true, text: 'Valid cells with ≥ 80% similarity' }, { cls: 'n', node: pctCell(site.share80) }]));
  sr.appendChild(row([{ h: true, text: 'Terrain similarity' }, { cls: 'n', node: pctCell(T.similarity * 100) }]));
  sr.appendChild(row([{ h: true, text: 'Terrain rank among regions and zones' }, { cls: 'n', text: String(T.rank) }]));
  sr.appendChild(row([{ h: true, text: 'No-data baseline (similarity of a surface without data)' }, { cls: 'n', node: pctCell(site.baseline) }]));
  sr.appendChild(row([{ h: true, text: 'Gain over the baseline' }, { cls: 'n', text: (site.gain >= 0 ? '+' : '') + f(site.gain) + '%' }]));

  // ---------- Observations ----------
  var notes = $('notes');
  var items = site.notes.slice();
  items.push('Mineral group totals are automatic: ' + site.emit.flagged + ' of ' +
             site.emit.minerals + ' EMIT minerals are flagged for review.');
  items.forEach(function (t) { notes.appendChild(el('li', null, t)); });

  // ---------- Mars signals ----------
  D.marsSignals.forEach(function (s) {
    $('marsRows').appendChild(row([
      { cls: 'muted', text: s.source },
      { h: true, text: s.mineral },
      { node: tag(s.level) },
      { cls: 'muted', text: s.unit },
      { cls: 'n', text: s.value >= 10 ? s.value.toFixed(2) : s.value.toFixed(4) }
    ]));
  });

  // ---------- Animate every bar on load ----------
  requestAnimationFrame(function () {
    requestAnimationFrame(function () {
      bars.forEach(function (b) { b.el.style.width = b.pct.toFixed(2) + '%'; });
    });
  });
})();
