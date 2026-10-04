// Draw translations over a page image.
//
// Each region's OCR box is covered with the median color of a thin band just
// outside it (white for most bubbles, the panel color for narration). The
// English then goes in the largest font that fits, in the biggest centered
// rectangle that is still empty bubble: vertical Chinese leaves tall, narrow
// boxes, and the bubble around them is usually much wider. Only when that
// isn't enough is the box widened over the art.
var ATX = globalThis.ATX || (globalThis.ATX = {});

(() => {
  const RING = 4; // px of the band sampled for the fill color
  const MAX_DIRTY = 0.01; // share of non-bubble pixels allowed inside the text rectangle

  // -> {blob, areas}: the rendered page, and per region (by index into
  // result.regions) the area it covers on the page, for hit-testing clicks.
  // opts.highlight marks bubbles the server flagged as possibly wrong.
  ATX.render = async function render(imageB64, type, result, opts = {}) {
    const blob = await (await fetch(`data:${type};base64,${imageB64}`)).blob();
    const bmp = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    bmp.close();

    const shown = result.regions.map((r, i) => ({ r, i })).filter(({ r }) => r.dst && r.dst.trim() && r.dst !== r.src);
    const regions = shown.map(({ r }) => r);
    // Sample every fill first: a box drawn earlier could cover a later one's band.
    const boxes = regions.map((r) => padded(r.box, canvas));
    const fills = boxes.map((b) => bandMedian(ctx, b, canvas));
    // Text may spread into empty bubble around its box, but never into another
    // region's box or a text area already placed.
    const placed = [];
    const areas = {};
    regions.forEach((r, i) => {
      const neighbours = boxes.filter((_, j) => j !== i).map((b) => halfGap(boxes[i], b));
      const snug = drawRegion(ctx, r, fills[i], canvas, [...neighbours, ...placed]);
      placed.push(snug);
      const b = boxes[i];
      areas[shown[i].i] = [Math.min(b[0], snug[0]), Math.min(b[1], snug[1]), Math.max(b[2], snug[2]), Math.max(b[3], snug[3])];
      if (opts.highlight && r.flagged) drawBadge(ctx, snug, canvas);
    });
    // toBlob / convertToBlob encode in idle time and took a flat ~1 s per page;
    // the synchronous toDataURL takes ~20 ms.
    return { blob: await (await fetch(canvas.toDataURL("image/jpeg", 0.92))).blob(), areas };
  };

  // A small amber "?" on the top-right corner of a text box: "this one may be wrong".
  function drawBadge(ctx, [x0, y0, x1], canvas) {
    const r = Math.max(9, Math.round(canvas.width * 0.016));
    const cx = Math.min(canvas.width - r - 1, x1), cy = Math.max(r + 1, y0);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = "#f2a516";
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, r / 6);
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.fillStyle = "#1b1b1b";
    ctx.font = `700 ${Math.round(r * 1.35)}px "Segoe UI", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("?", cx, cy + r * 0.08);
  }

  ATX.renderInternals = { bandMedian, layout, padded }; // for dev/harness.html debugging

  // A neighbouring box grown halfway across the gap toward `own`, so two regions
  // split the empty space between them instead of the first one taking it all.
  function halfGap(own, other) {
    const [ax0, ay0, ax1, ay1] = own;
    let [bx0, by0, bx1, by1] = other;
    if (bx0 < ax1 && bx1 > ax0) { // stacked vertically
      if (by0 >= ay1) by0 -= Math.floor((by0 - ay1) / 2);
      else if (by1 <= ay0) by1 += Math.floor((ay0 - by1) / 2);
    }
    if (by0 < ay1 && by1 > ay0) { // side by side
      if (bx0 >= ax1) bx0 -= Math.floor((bx0 - ax1) / 2);
      else if (bx1 <= ax0) bx1 += Math.floor((ax0 - bx1) / 2);
    }
    return [bx0, by0, bx1, by1];
  }

  function padded([x0, y0, x1, y1], { width, height }) {
    const p = ATX.config.padPx;
    return [Math.max(0, x0 - p), Math.max(0, y0 - p), Math.min(width, x1 + p), Math.min(height, y1 + p)];
  }

  // Median color of the band just outside the box (per channel, via histograms).
  function bandMedian(ctx, [x0, y0, x1, y1], { width, height }) {
    const bx0 = Math.max(0, x0 - RING), by0 = Math.max(0, y0 - RING);
    const bx1 = Math.min(width, x1 + RING), by1 = Math.min(height, y1 + RING);
    const data = ctx.getImageData(bx0, by0, bx1 - bx0, by1 - by0).data;
    const hist = [new Uint32Array(256), new Uint32Array(256), new Uint32Array(256)];
    let n = 0;
    const w = bx1 - bx0;
    for (let y = by0; y < by1; y++) {
      for (let x = bx0; x < bx1; x++) {
        if (x >= x0 && x < x1 && y >= y0 && y < y1) continue; // inside the box
        const i = ((y - by0) * w + (x - bx0)) * 4;
        hist[0][data[i]]++; hist[1][data[i + 1]]++; hist[2][data[i + 2]]++;
        n++;
      }
    }
    if (!n) return [255, 255, 255];
    return hist.map((h) => {
      let acc = 0;
      for (let v = 0; v < 256; v++) if ((acc += h[v]) * 2 >= n) return v;
      return 255;
    });
  }

  function drawRegion(ctx, region, fill, canvas, others) {
    const C = ATX.config;
    const box = padded(region.box, canvas);
    const lay = layout(ctx, region.dst, box, fill, canvas, others, region.frame);

    // Paint only the original Chinese and a snug box around the English, not the
    // whole area the layout searched: that area is only mostly empty.
    ctx.font = font(lay.size);
    const lineH = lay.size * C.lineHeight;
    const textW = Math.max(...lay.lines.map((l) => ctx.measureText(l).width));
    const textH = lay.lines.length * lineH;
    const [lx0, ly0, lx1, ly1] = lay.box;
    const cx = (lx0 + lx1) / 2, cy = (ly0 + ly1) / 2;
    const snug = [
      Math.max(lx0, Math.floor(cx - textW / 2 - C.padPx)), Math.max(ly0, Math.floor(cy - textH / 2 - C.padPx / 2)),
      Math.min(lx1, Math.ceil(cx + textW / 2 + C.padPx)), Math.min(ly1, Math.ceil(cy + textH / 2 + C.padPx / 2)),
    ];

    ctx.fillStyle = `rgb(${fill.join(",")})`;
    for (const [x0, y0, x1, y1] of [box, snug]) {
      ctx.beginPath();
      ctx.roundRect(x0, y0, x1 - x0, y1 - y0, Math.min(8, lay.size / 2));
      ctx.fill();
    }

    const lum = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2];
    ctx.fillStyle = lum > 128 ? "#111" : "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    let y = cy - ((lay.lines.length - 1) * lineH) / 2;
    for (const line of lay.lines) {
      ctx.fillText(line, cx, y);
      y += lineH;
    }
    return snug; // later regions may use the rest of the searched area
  }

  function font(size) {
    return `${ATX.config.fontWeight} ${size}px ${ATX.config.font}`;
  }

  // -> {size, lines, box}: where and how big to set the text.
  function layout(ctx, text, box, fill, canvas, others = [], frame = null) {
    const inBubble = layoutInBubble(ctx, text, box, fill, canvas, others, frame);
    if (inBubble.size >= ATX.config.comfortableFontPx) return inBubble;
    const widened = layoutWidened(ctx, text, box, canvas.width);
    const [wx0, wy0, wx1, wy1] = widened.box;
    const overlaps = others.some(([ox0, oy0, ox1, oy1]) => ox0 < wx1 && ox1 > wx0 && oy0 < wy1 && oy1 > wy0);
    return widened.size > inBubble.size && !overlaps ? widened : inBubble;
  }

  // Try rectangles centered on the OCR box: for each width, the tallest one that
  // is still empty bubble (fill-colored, or inside the OCR box, and not inside
  // another region's box). Keep the one allowing the largest font.
  function layoutInBubble(ctx, text, box, fill, { width: W, height: H }, others, frame) {
    const C = ATX.config;
    const [x0, y0, x1, y1] = box;
    const w = x1 - x0, h = y1 - y0;
    // Search inside the detected speech bubble when the server found one;
    // otherwise around the text, by a margin based on its size.
    const [ax0, ay0, ax1, ay1] = frame
      ? [Math.max(0, Math.min(frame[0], x0)), Math.max(0, Math.min(frame[1], y0)),
         Math.min(W, Math.max(frame[2], x1)), Math.min(H, Math.max(frame[3], y1))]
      : [Math.max(0, x0 - Math.max(w, 60)), Math.max(0, y0 - Math.round(h * 0.3)),
         Math.min(W, x1 + Math.max(w, 60)), Math.min(H, y1 + Math.round(h * 0.3))];
    const aw = ax1 - ax0, ah = ay1 - ay0;
    const data = ctx.getImageData(ax0, ay0, aw, ah).data;

    // Summed-area table of "dirty" pixels: neither fill-colored nor inside the box.
    // Pixels in other regions' boxes are dirty no matter their color.
    const near = others.filter(([ox0, oy0, ox1, oy1]) => ox0 < ax1 && ox1 > ax0 && oy0 < ay1 && oy1 > ay0);
    const blocked = (px, py) => near.some(([ox0, oy0, ox1, oy1]) => px >= ox0 && px < ox1 && py >= oy0 && py < oy1);
    const sat = new Uint32Array((aw + 1) * (ah + 1));
    for (let y = 0; y < ah; y++) {
      let row = 0;
      for (let x = 0; x < aw; x++) {
        const px = ax0 + x, py = ay0 + y;
        let dirty = 0;
        if (px >= x0 && px < x1 && py >= y0 && py < y1) {
          dirty = 0; // its own box: covered by the fill anyway
        } else if (near.length && blocked(px, py)) {
          dirty = 1000; // rules out any rectangle that touches it
        } else {
          const i = (y * aw + x) * 4;
          dirty = Math.abs(data[i] - fill[0]) + Math.abs(data[i + 1] - fill[1]) + Math.abs(data[i + 2] - fill[2]) >= 60 ? 1 : 0;
        }
        row += dirty;
        sat[(y + 1) * (aw + 1) + x + 1] = sat[y * (aw + 1) + x + 1] + row;
      }
    }
    const dirtyIn = (rx0, ry0, rx1, ry1) => { // area coordinates, exclusive end
      const s = aw + 1;
      return sat[ry1 * s + rx1] - sat[ry0 * s + rx1] - sat[ry1 * s + rx0] + sat[ry0 * s + rx0];
    };

    // Center on the bubble when known (its widest, tallest part), else on the text.
    const [fx0, fy0, fx1, fy1] = frame || box;
    const cx = (fx0 + fx1) / 2 - ax0, cy = (fy0 + fy1) / 2 - ay0;
    let best = null;
    const step = Math.max(4, Math.round(w * 0.1));
    for (let rw = w; rw <= aw; rw += step) {
      const rx0 = Math.round(cx - rw / 2), rx1 = rx0 + rw;
      if (rx0 < 0 || rx1 > aw) break;
      // Tallest centered rectangle at this width that's still clean (grows monotonically).
      let lo = 0, hi = Math.floor(Math.min(cy, ah - cy) * 2);
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        const ry0 = Math.round(cy - mid / 2), ry1 = ry0 + mid;
        if (dirtyIn(rx0, ry0, rx1, ry1) <= MAX_DIRTY * rw * mid) lo = mid; else hi = mid - 1;
      }
      if (lo < C.minFontPx * 1.5) continue;
      const fitted = largestFont(ctx, text, rw - 2 * C.padPx, lo - 2 * C.padPx);
      if (!best || fitted.size > best.size) {
        const ry0 = Math.round(cy - lo / 2);
        best = { ...fitted, box: [ax0 + rx0, ay0 + ry0, ax0 + rx1, ay0 + ry0 + lo] };
      }
    }
    return best || { ...largestFont(ctx, text, w - 2 * C.padPx, h - 2 * C.padPx), box };
  }

  // Fallback: widen the box (centered, inside the image), covering art.
  function layoutWidened(ctx, text, box, imageWidth) {
    const C = ATX.config;
    const [x0, y0, x1, y1] = box;
    const w = x1 - x0, h = y1 - y0;
    let best = null;
    for (const scale of [1, 1.3, 1.6, 2, 2.5, 3]) {
      const width = Math.min(imageWidth - 4, Math.round(w * scale));
      if (best && width <= best.box[2] - best.box[0]) break; // can't widen further
      const nx0 = Math.max(2, Math.min(Math.round((x0 + x1) / 2 - width / 2), imageWidth - 2 - width));
      const fitted = largestFont(ctx, text, width - 2 * C.padPx, h - 2 * C.padPx);
      if (!best || fitted.size > best.size) best = { ...fitted, box: [nx0, y0, nx0 + width, y1] };
      if (best.size >= C.comfortableFontPx) break;
    }
    return best;
  }

  function largestFont(ctx, text, maxW, maxH) {
    const C = ATX.config;
    for (let size = C.maxFontPx; size >= C.minFontPx; size--) {
      ctx.font = font(size);
      const { lines, broke } = wrap(ctx, text, maxW);
      // Splitting a word only counts as fitting at the smallest size; before
      // that, a smaller font (or a wider box) is better.
      if (!broke && lines.length * size * C.lineHeight <= maxH) return { size, lines };
    }
    ctx.font = font(C.minFontPx);
    return { size: C.minFontPx, lines: wrap(ctx, text, maxW).lines }; // overflows; the best we can do
  }

  function wrap(ctx, text, maxW) {
    const lines = [];
    let line = "", broke = false;
    for (const word of text.split(/\s+/).filter(Boolean)) {
      const test = line ? `${line} ${word}` : word;
      if (ctx.measureText(test).width <= maxW) {
        line = test;
        continue;
      }
      if (line) lines.push(line);
      line = word;
      while (ctx.measureText(line).width > maxW && line.length > 1) {
        // A single word wider than the box: break it.
        let cut = line.length - 1;
        while (cut > 1 && ctx.measureText(line.slice(0, cut) + "-").width > maxW) cut--;
        lines.push(line.slice(0, cut) + "-");
        line = line.slice(cut);
        broke = true;
      }
    }
    if (line) lines.push(line);
    return { lines, broke };
  }
})();
