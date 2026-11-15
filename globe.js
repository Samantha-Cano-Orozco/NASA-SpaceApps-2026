/* Canvas globe without dependencies.
 * Orthographic projection of an equirectangular texture (longitude -180 at the left edge).
 *
 * createGlobe({
 *   canvas, texture, lat, lon, zoom,
 *   zones:   [{ id, lat, lon, radius (deg), color }],   // areas; drawn with a visible ring
 *   markers: [{ id, lat, lon, color }],                  // points
 *   interactive, onSelect(item), onHover(item | null), onZoom(z)
 * })
 * item: { id, kind: 'zone' | 'point' }
 */
(function () {
  'use strict';

  var D2R = Math.PI / 180;
  var R2D = 180 / Math.PI;
  var MIN_ZOOM = 1;
  var MIN_RING_PX = 10;   // zones never draw smaller than this (CSS px), so they stay visible from orbit
  var TINT_MIN_PX = 5;    // zones get a shaded area only when their true size is at least this

  function unit(lat, lon) {
    var p = lat * D2R, l = lon * D2R, c = Math.cos(p);
    return [c * Math.cos(l), c * Math.sin(l), Math.sin(p)];
  }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function hexRGB(h) {
    var n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  function loadTexture(src, done) {
    var img = new Image();
    img.onload = function () {
      var c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      var x = c.getContext('2d');
      x.drawImage(img, 0, 0);
      done({ w: c.width, h: c.height, data: x.getImageData(0, 0, c.width, c.height).data });
    };
    img.src = src;
  }

  function createGlobe(opts) {
    var canvas = opts.canvas;
    var ctx = canvas.getContext('2d');
    var interactive = opts.interactive !== false;
    var zones = (opts.zones || []).map(function (z) {
      return { id: z.id, kind: 'zone', lat: z.lat, lon: z.lon, radius: z.radius,
               rgb: hexRGB(z.color || '#2d5f8b'), center: unit(z.lat, z.lon), active: false, hover: false };
    });
    var markers = (opts.markers || []).map(function (m) {
      return { id: m.id, kind: 'point', lat: m.lat, lon: m.lon, rgb: hexRGB(m.color || '#2d5f8b'),
               unit: unit(m.lat, m.lon), active: false, hover: false, _px: null };
    });
    var state = { lat: opts.lat || 0, lon: opts.lon || 0, zoom: opts.zoom || MIN_ZOOM };
    var maxZoom = opts.maxZoom || 30;
    var tex = null;
    var box = { size: 0, dpr: 1, r: 0 };
    var drag = null, moved = 0, frameId = 0, tweenId = 0;

    canvas.style.touchAction = 'none';

    function basis() {
      var p = state.lat * D2R, l = state.lon * D2R;
      return {
        C: [Math.cos(p) * Math.cos(l), Math.cos(p) * Math.sin(l), Math.sin(p)],
        E: [-Math.sin(l), Math.cos(l), 0],
        N: [-Math.sin(p) * Math.cos(l), -Math.sin(p) * Math.sin(l), Math.cos(p)]
      };
    }

    function radiusPx() { return box.r * state.zoom; }

    function resize() {
      var css = canvas.clientWidth || 300;
      var dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(css * dpr);
      canvas.height = canvas.width;
      box.size = canvas.width;
      box.dpr = dpr;
      box.r = canvas.width * 0.5 * 0.96;
      requestDraw();
    }

    function paint(b) {
      var S = box.size, r = radiusPx(), cx = S / 2, cy = S / 2;
      var E = b.E, N = b.N, C = b.C;
      var img = ctx.createImageData(S, S);
      var out = img.data;
      var W = tex.w, H = tex.h, src = tex.data;
      var band = 2 * box.dpr / r; // outline thickness in radians
      var zs = [];
      zones.forEach(function (z) {
        var rad = z.radius * D2R;
        var truePx = rad * r / box.dpr;       // approximate on-screen radius in CSS px
        if (truePx < TINT_MIN_PX) return;     // too small to shade; the ring marks it
        zs.push({ c: z.center, rgb: z.rgb,
                  cOut: Math.cos(rad + band), cIn: Math.cos(Math.max(0, rad - band)),
                  alpha: z.hover ? 0.5 : z.active ? 0.4 : 0.26 });
      });
      for (var py = 0; py < S; py++) {
        var sy = -(py + 0.5 - cy) / r;
        if (Math.abs(sy) > 1) continue;
        for (var px = 0; px < S; px++) {
          var sx = (px + 0.5 - cx) / r;
          var d = sx * sx + sy * sy;
          if (d > 1) continue;
          var sz = Math.sqrt(1 - d);
          var wx = sx * E[0] + sy * N[0] + sz * C[0];
          var wy = sx * E[1] + sy * N[1] + sz * C[1];
          var wz = sx * E[2] + sy * N[2] + sz * C[2];
          var lat = Math.asin(clamp(wz, -1, 1)) * R2D;
          var lon = Math.atan2(wy, wx) * R2D;
          var u = clamp(Math.floor((lon + 180) / 360 * W), 0, W - 1);
          var v = clamp(Math.floor((90 - lat) / 180 * H), 0, H - 1);
          var si = (v * W + u) * 4;
          var oi = (py * S + px) * 4;
          var k = 0.4 + 0.6 * Math.pow(sz, 0.5);
          var R = src[si] * k, G = src[si + 1] * k, B = src[si + 2] * k;
          for (var j = 0; j < zs.length; j++) {
            var z = zs[j];
            var dt = wx * z.c[0] + wy * z.c[1] + wz * z.c[2];
            if (dt < z.cOut) continue;
            if (dt <= z.cIn) {                       // outline
              R = z.rgb[0]; G = z.rgb[1]; B = z.rgb[2];
            } else {                                 // inside
              var a = z.alpha;
              R = R * (1 - a) + z.rgb[0] * a;
              G = G * (1 - a) + z.rgb[1] * a;
              B = B * (1 - a) + z.rgb[2] * a;
            }
          }
          out[oi] = R; out[oi + 1] = G; out[oi + 2] = B; out[oi + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }

    function screenOf(vec, b) {
      var S = box.size, r = radiusPx(), cx = S / 2, cy = S / 2;
      return [cx + dot(vec, b.E) * r, cy - dot(vec, b.N) * r, dot(vec, b.C)];
    }

    function drawRing(X, Y, px, rgb, on) {
      var dpr = box.dpr;
      var c = 'rgb(' + rgb.join(',') + ')';
      ctx.beginPath();
      ctx.arc(X, Y, px, 0, Math.PI * 2);
      ctx.lineWidth = (on ? 2.4 : 1.6) * dpr;
      ctx.strokeStyle = c;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(X, Y, (on ? 3 : 2.2) * dpr, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.fill();
    }

    function drawOverlays(b) {
      var dpr = box.dpr;
      var r = radiusPx();
      zones.forEach(function (z) {
        var s = screenOf(z.center, b);
        z._px = null;
        if (s[2] <= 0.02) return;
        var truePx = z.radius * D2R * r;
        var ringPx = Math.max(truePx, MIN_RING_PX * dpr);
        z._px = [s[0], s[1], ringPx];
        drawRing(s[0], s[1], ringPx, z.rgb, z.active || z.hover);
      });
      markers.forEach(function (m) {
        var s = screenOf(m.unit, b);
        m._px = null;
        if (s[2] <= 0.02) return;
        m._px = [s[0], s[1]];
        drawRing(s[0], s[1], (m.active || m.hover ? 8 : 6) * dpr, m.rgb, m.active || m.hover);
      });
    }

    function drawPlaceholder() {
      var S = box.size;
      ctx.clearRect(0, 0, S, S);
    }

    function draw() {
      frameId = 0;
      if (!tex) { drawPlaceholder(); return; }
      var b = basis();
      paint(b);
      drawOverlays(b);
    }

    function requestDraw() {
      if (!frameId) frameId = requestAnimationFrame(draw);
    }

    function canvasToDevice(e) {
      var rect = canvas.getBoundingClientRect();
      var scale = canvas.width / rect.width;
      return [(e.clientX - rect.left) * scale, (e.clientY - rect.top) * scale];
    }

    function pixelToWorld(X, Y) {
      var S = box.size, r = radiusPx(), b = basis();
      var sx = (X - S / 2) / r, sy = -(Y - S / 2) / r;
      var d = sx * sx + sy * sy;
      if (d > 1) return null;
      var sz = Math.sqrt(1 - d);
      return [sx * b.E[0] + sy * b.N[0] + sz * b.C[0],
              sx * b.E[1] + sy * b.N[1] + sz * b.C[1],
              sx * b.E[2] + sy * b.N[2] + sz * b.C[2]];
    }

    function hitAt(p) {
      var dpr = box.dpr, best = null, bestD = 1e9;
      markers.forEach(function (m) {
        if (!m._px) return;
        var d = Math.hypot(m._px[0] - p[0], m._px[1] - p[1]);
        if (d < 14 * dpr && d < bestD) { bestD = d; best = m; }
      });
      if (best) return best;
      zones.forEach(function (z) {
        if (!z._px) return;
        var d = Math.hypot(z._px[0] - p[0], z._px[1] - p[1]);
        if (d <= z._px[2] + 6 * dpr && d < bestD) { bestD = d; best = z; }
      });
      if (best) return best;
      var w = pixelToWorld(p[0], p[1]);
      if (!w) return null;
      zones.forEach(function (z) {
        if (dot(w, z.center) >= Math.cos(z.radius * D2R) && !best) best = z;
      });
      return best;
    }

    function describe(item) {
      return item ? { id: item.id, kind: item.kind } : null;
    }

    function setHover(item) {
      zones.forEach(function (z) { z.hover = z === item; });
      markers.forEach(function (m) { m.hover = m === item; });
      canvas.style.cursor = item ? 'pointer' : (interactive ? 'grab' : 'default');
      requestDraw();
      if (opts.onHover) opts.onHover(describe(item));
    }

    function wrapLon(l) { return ((l + 540) % 360) - 180; }

    function cancelTween() {
      if (tweenId) { cancelAnimationFrame(tweenId); tweenId = 0; }
    }

    function zoomTo(z) {
      state.zoom = clamp(z, MIN_ZOOM, maxZoom);
      requestDraw();
      if (opts.onZoom) opts.onZoom(state.zoom);
    }

    if (interactive) {
      canvas.style.cursor = 'grab';
      canvas.addEventListener('pointerdown', function (e) {
        cancelTween();
        drag = { x: e.clientX, y: e.clientY };
        moved = 0;
        canvas.setPointerCapture(e.pointerId);
      });
      canvas.addEventListener('pointermove', function (e) {
        if (drag) {
          var dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          drag.x = e.clientX; drag.y = e.clientY;
          moved += Math.abs(dx) + Math.abs(dy);
          var rCss = radiusPx() / box.dpr;
          state.lon = wrapLon(state.lon - dx * 90 / rCss);
          state.lat = clamp(state.lat + dy * 90 / rCss, -85, 85);
          requestDraw();
          return;
        }
        var hit = hitAt(canvasToDevice(e));
        var current = zones.find(function (z) { return z.hover; }) || markers.find(function (m) { return m.hover; }) || null;
        if (hit !== current) setHover(hit);
      });
      canvas.addEventListener('pointerup', function (e) {
        var click = drag && moved < 5;
        drag = null;
        if (click) {
          var hit = hitAt(canvasToDevice(e));
          if (hit && opts.onSelect) opts.onSelect(describe(hit));
        }
      });
      canvas.addEventListener('pointerleave', function () { if (!drag) setHover(null); });
      // Wheel zooms only with Ctrl held, so plain scrolling moves between the full-height worlds.
      canvas.addEventListener('wheel', function (e) {
        if (!e.ctrlKey) return;
        e.preventDefault();
        zoomTo(state.zoom * (e.deltaY < 0 ? 1.25 : 0.8));
      }, { passive: false });
    }

    // Animated move to a coordinate (shortest longitude path), with zoom.
    function focus(lat, lon, zoom, duration) {
      cancelTween();
      var from = { lat: state.lat, lon: state.lon, z: Math.log(state.zoom) };
      var dLon = wrapLon(lon - from.lon);
      var toZ = Math.log(clamp(zoom || state.zoom, MIN_ZOOM, maxZoom));
      var t0 = null, dur = duration || 800;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var t = Math.min(1, (ts - t0) / dur);
        var e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        state.lat = from.lat + (lat - from.lat) * e;
        state.lon = wrapLon(from.lon + dLon * e);
        state.zoom = Math.exp(from.z + (toZ - from.z) * e);
        draw();
        tweenId = t < 1 ? requestAnimationFrame(step) : 0;
      }
      tweenId = requestAnimationFrame(step);
    }

    function setActive(id) {
      zones.forEach(function (z) { z.active = z.id === id; });
      markers.forEach(function (m) { m.active = m.id === id; });
      requestDraw();
    }

    function setEnabled(on) {
      canvas.style.pointerEvents = on ? 'auto' : 'none';
    }

    loadTexture(opts.texture, function (t) { tex = t; requestDraw(); });
    window.addEventListener('resize', resize);
    resize();

    return {
      focus: focus,
      setActive: setActive,
      setEnabled: setEnabled,
      zoomIn: function () { zoomTo(state.zoom * 1.5); },
      zoomOut: function () { zoomTo(state.zoom / 1.5); },
      resize: resize
    };
  }

  window.createGlobe = createGlobe;
})();
