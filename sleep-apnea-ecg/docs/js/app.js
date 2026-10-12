/* Renders every section from data/results.json and runs the demo from data/model.json. */
(function () {
  "use strict";

  const SVGNS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 2) => (v == null ? "n/a" : Number(v).toFixed(d));
  const pct = (v, d = 1) => (v == null ? "n/a" : (v * 100).toFixed(d) + "%");
  const int = (v) => (v == null ? "n/a" : Number(v).toLocaleString("en-US"));
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : document.createTextNode(String(k)));
    return n;
  }
  function s(tag, attrs = {}, text) {
    const n = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  }
  const ticks = (lo, hi, n) => {
    const step = niceStep((hi - lo) / n);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const m = raw / p;
    return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  }

  async function loadJSON(url) {
    try {
      const r = await fetch(url, { cache: "no-cache" });
      if (!r.ok) return null;
      return await r.json();
    } catch (_) {
      return null;
    }
  }

  // ------------------------------------------------------------------ generic line chart
  function lineChart(container, cfg) {
    const W = cfg.width || 900, H = cfg.height || 300;
    const m = { l: 52, r: 16, t: 14, b: 40, ...(cfg.margin || {}) };
    const [x0, x1] = cfg.x, [y0, y1] = cfg.y;
    const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": cfg.label || "" });

    for (const b of cfg.bands || []) {
      svg.append(s("rect", { x: X(b.from), y: m.t, width: Math.max(0, X(b.to) - X(b.from)), height: H - m.t - m.b, class: b.class }));
      if (b.text) svg.append(s("text", { x: X(b.from) + 4, y: m.t + 13, class: "band-label " + b.textClass }, b.text));
    }
    for (const t of cfg.yTicks || ticks(y0, y1, 4)) {
      svg.append(s("line", { x1: m.l - 4, x2: m.l, y1: Y(t), y2: Y(t), class: "ax" }));
      svg.append(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", class: "tick" }, cfg.yFmt ? cfg.yFmt(t) : t));
    }
    for (const t of cfg.xTicks || ticks(x0, x1, 6)) {
      svg.append(s("line", { x1: X(t), x2: X(t), y1: H - m.b, y2: H - m.b + 4, class: "ax" }));
      svg.append(s("text", { x: X(t), y: H - m.b + 17, "text-anchor": "middle", class: "tick" }, cfg.xFmt ? cfg.xFmt(t) : t));
    }
    svg.append(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "ax" }));
    svg.append(s("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "ax" }));
    if (cfg.xTitle) svg.append(s("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "ax-title" }, cfg.xTitle));
    if (cfg.yTitle) svg.append(s("text", { x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", class: "ax-title" }, cfg.yTitle));
    for (const extra of cfg.extras || []) svg.append(extra({ X, Y, m, W, H }));

    for (const ser of cfg.series) {
      const d = ser.points.map(([a, b], i) => `${i ? "L" : "M"}${X(a).toFixed(1)},${Y(b).toFixed(1)}`).join("");
      svg.append(s("path", { d, class: "line " + (ser.class || "") }));
      if (ser.dots) for (const [a, b] of ser.points) svg.append(s("circle", { cx: X(a), cy: Y(b), r: 2, class: "dot" }));
    }
    if (cfg.legend) {
      let lx = m.l + 12;
      const ly = m.t + 6;
      cfg.legend.forEach((g) => {
        svg.append(s("line", { x1: lx, x2: lx + 22, y1: ly, y2: ly, class: "line " + (g.class || "") }));
        const t = s("text", { x: lx + 28, y: ly + 4, class: "legend" }, g.text);
        svg.append(t);
        lx += 40 + g.text.length * 6.6;
      });
    }
    container.replaceChildren(svg);
    return svg;
  }

  // ------------------------------------------------------------------ hero ECG trace
  // PQRST template (sum of Gaussians, in mV). Timing between beats comes from real RR data.
  const WAVES = [[0.12, -0.2, 0.028], [-0.1, -0.035, 0.01], [1.15, 0, 0.011], [-0.28, 0.032, 0.012], [0.3, 0.25, 0.045]];
  function ecgValue(t, beats) {
    let v = 0;
    for (const b of beats) {
      const dt = t - b;
      if (dt < -0.35 || dt > 0.45) continue;
      for (const [a, mu, sd] of WAVES) v += a * Math.exp(-((dt - mu) ** 2) / (2 * sd * sd));
    }
    return v;
  }

  function drawHero(rr) {
    const svg = $("hero-trace");
    const box = svg.getBoundingClientRect();
    const W = Math.max(320, box.width), H = box.height;
    const pxPerSec = 125, pxPerMv = 50;            // 25 mm/s and 10 mm/mV at 5 px per mm
    const secs = W / pxPerSec;
    const baseline = H * 0.66;
    const beats = [];
    let t = 0.55;
    for (let i = 0; t < secs + 1 && i < rr.length * 4; i++) { beats.push(t); t += rr[i % rr.length]; }
    const step = 1 / 250;
    let d = "";
    for (let x = 0; x <= secs; x += step) {
      const y = baseline - ecgValue(x, beats) * pxPerMv;
      d += (x ? "L" : "M") + (x * pxPerSec).toFixed(1) + "," + y.toFixed(1);
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();
    const path = s("path", { d });
    svg.append(path);

    // annotate one RR interval
    if (beats.length > 3) {
      const a = beats[1] * pxPerSec, b = beats[2] * pxPerSec, y = baseline - 1.32 * pxPerMv;
      svg.append(s("line", { x1: a, x2: b, y1: y, y2: y, class: "rr-span" }));
      svg.append(s("line", { x1: a, x2: a, y1: y - 5, y2: y + 5, class: "rr-span" }));
      svg.append(s("line", { x1: b, x2: b, y1: y - 5, y2: y + 5, class: "rr-span" }));
      svg.append(s("text", { x: (a + b) / 2, y: y - 9, "text-anchor": "middle", class: "rr-text" }, `RR ${(beats[2] - beats[1]).toFixed(2)} s`));
    }
    if (!reduceMotion && !drawHero.done) {
      const len = path.getTotalLength();
      path.style.strokeDasharray = len;
      path.style.strokeDashoffset = len;
      path.getBoundingClientRect();
      path.style.transition = "stroke-dashoffset 2.6s cubic-bezier(.45,.05,.35,1)";
      path.style.strokeDashoffset = "0";
      drawHero.done = true;
    }
  }

  // ------------------------------------------------------------------ RR tachogram
  function tachogram(container, minutes, opts = {}) {
    let t = 0;
    const pts = [], bands = [];
    for (const m of minutes) {
      const start = t;
      for (const v of m.rr) { t += v; pts.push([t, v]); }
      if (m.label != null) {
        const ap = m.label === 1;
        bands.push({ from: start, to: t, class: ap ? "band-apnea" : "band-normal",
          text: opts.bandText === false ? null : (ap ? "A" : "N"), textClass: ap ? "apnea" : "normal" });
      }
    }
    const ys = pts.map((p) => p[1]);
    const lo = Math.max(0.25, Math.min(...ys) - 0.05), hi = Math.min(2.05, Math.max(...ys) + 0.05);
    const long = t > 150;
    lineChart(container, {
      width: opts.width || 1000, height: opts.height || 280,
      x: [0, t], y: [lo, hi],
      xTicks: long ? ticks(0, t / 60, 8).map((v) => v * 60) : undefined,
      xFmt: long ? (v) => `${Math.round(v / 60) + (opts.startMinute || 0)}` : (v) => `${v}`,
      xTitle: long ? "Minutes" : "Seconds",
      yTitle: "RR interval (s)",
      yFmt: (v) => v.toFixed(2),
      bands,
      series: [{ points: pts, dots: !long }],
      label: opts.label || "RR interval over time",
    });
  }

  // ------------------------------------------------------------------ confusion matrix
  function confusion(container, cm) {
    if (!cm) {
      container.replaceChildren(el("div", { class: "cm-empty" },
        "The original notebook did not print a confusion matrix for this model. Run the pipeline to fill it in."));
      return;
    }
    const [[tn, fp], [fn, tp]] = cm;
    const rows = [tn + fp, fn + tp];
    const cell = (v, row, ok) => {
      const share = rows[row] ? v / rows[row] : 0;
      const tint = ok ? `color-mix(in srgb, var(--normal) ${Math.round(share * 22)}%, transparent)`
                      : `color-mix(in srgb, var(--apnea) ${Math.round(share * 30)}%, transparent)`;
      return el("div", { class: "cell " + (ok ? "ok" : "bad"), style: `background:${tint}` },
        el("b", {}, int(v)), el("span", {}, pct(share, 0) + " of row"));
    };
    container.replaceChildren(el("div", { class: "cm", role: "table", "aria-label": "Confusion matrix" },
      el("div"), el("div", { class: "h" }, "Predicted normal"), el("div", { class: "h" }, "Predicted apnea"),
      el("div", { class: "rh" }, "Actually normal"), cell(tn, 0, true), cell(fp, 0, false),
      el("div", { class: "rh" }, "Actually apnea"), cell(fn, 1, false), cell(tp, 1, true)));
  }

  function statList(container, items) {
    container.replaceChildren(...items.map(([k, v, sub, cls]) =>
      el("div", {}, el("dt", {}, k), el("dd", { class: cls || null }, v, sub ? el("small", {}, sub) : null))));
  }

  // ------------------------------------------------------------------ sections
  function renderBanner(R) {
    const b = $("source-banner");
    if (R.source === "pipeline" && !R.quick) return;
    b.hidden = false;
    b.textContent = R.source === "pipeline"
      ? "These numbers come from a --quick smoke-test run with very few epochs. Run the full pipeline before reading anything into them."
      : "Showing the numbers recorded in the original notebooks, before the fixes listed under Audit. Running scripts/run_pipeline.py replaces them with the corrected results, adds ROC curves and per-patient results, and switches on the live demo.";
  }

  function renderSignal(R) {
    const d = R.dataset || {};
    const st = d.sample_stretch;
    if (st && st.minutes && st.minutes.length > 3) {
      tachogram($("signal-chart"), st.minutes, { height: 300, startMinute: st.minutes[0].minute, label: "RR intervals of record a01 across the onset of apnea" });
      const first = st.minutes[0].minute, last = st.minutes[st.minutes.length - 1].minute;
      $("signal-caption").textContent = `Record a01, minutes ${first} to ${last + 1} of the night. Shaded minutes are labelled apnea by experts. Watch the RR interval start to swing in a regular cycle.`;
    } else if (d.sample_rr) {
      tachogram($("signal-chart"), [{ rr: d.sample_rr.rr, label: d.sample_rr.label }], { height: 260, bandText: false });
      $("signal-caption").textContent = `Record a01, minute ${d.sample_rr.minute}: one normal minute, ${d.sample_rr.rr.length} RR intervals. After a pipeline run this figure shows a longer stretch that crosses into apnea.`;
    }
  }

  function renderStage1(R) {
    const s1 = R.stage1;
    if (!s1) return;
    const t = s1.test;
    document.querySelectorAll('[data-k="s1.params"]').forEach((n) => (n.textContent = int(s1.params)));
    $("s1-windows").textContent = `${int(s1.windows)} complete windows, ${int(s1.labels)} labels`;
    $("s1-acc").textContent = pct(t.accuracy, 2);
    $("s1-classes").textContent = `${int(s1.apnea)} apnea minutes against ${int(s1.normal)} normal`;
    $("s1-base-inline").textContent = pct(t.baseline_accuracy, 0);
    const bars = [
      ["This network", t.accuracy, ""],
      ['Always "apnea"', t.baseline_accuracy, "trivial"],
      ["Normal minutes caught", t.normal.recall, "trivial"],
    ];
    $("s1-baseline").replaceChildren(...bars.map(([k, v, c]) =>
      el("div", { class: "bl-row" }, el("span", {}, k),
        el("div", { class: "bl-track" }, el("div", { class: "bl-fill " + c, style: `width:${(v * 100).toFixed(1)}%` })),
        el("span", { class: "bl-val" }, pct(v, 0)))));
    confusion($("s1-cm"), t.cm);
    if (s1.history && s1.history.loss) {
      $("s1-history-fig").hidden = false;
      historyChart($("s1-history"), s1.history, "loss", "Loss");
    }
  }

  function historyChart(container, h, key, title, size = [900, 260]) {
    const tr = h[key] || [], va = h["val_" + key] || [];
    const all = tr.concat(va);
    const lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.08 || 0.05;
    lineChart(container, {
      width: size[0], height: size[1], x: [1, Math.max(2, tr.length)], y: [lo - pad, hi + pad],
      xTitle: "Epoch", yTitle: title, yFmt: (v) => v.toFixed(2),
      series: [{ points: tr.map((v, i) => [i + 1, v]) }, { points: va.map((v, i) => [i + 1, v]), class: "val" }],
      legend: [{ text: "Training", class: "" }, { text: "Validation", class: "val" }],
      label: `${title} per epoch`,
    });
  }

  function renderSplit(R) {
    const sp = R.split;
    if (!sp) return;
    const rows = [["train", "Training"], ["val", "Validation"], ["test", "Test"]];
    $("split-viz").replaceChildren(...rows.map(([k, name]) => {
      const c = sp.counts[k];
      return el("div", { class: "split-row " + k },
        el("div", { class: "split-name" }, el("b", {}, `${name}: ${c.records} records`),
          el("span", {}, `${int(c.n)} minutes, ${pct(c.apnea / c.n, 0)} apnea`)),
        el("div", { class: "tiles" }, [...sp[k]].sort().map((id) => el("span", { class: "tile", title: id }, id))));
    }));
  }

  function renderStage2(R) {
    const d = R.dataset, s2 = R.stage2;
    if (d) {
      statList($("s2-data"), [
        ["Records", int(d.records), "a01 to a20, b01 to b05, c01 to c10"],
        ["Usable minutes", int(d.segments), `${int(d.segments_raw - d.segments)} dropped: fewer than ${d.min_rr_per_minute} beats detected`],
        ["Normal minutes", int(d.normal), pct(d.normal / d.segments, 0)],
        ["Apnea minutes", int(d.apnea), pct(d.apnea / d.segments, 0), "apnea"],
        ["RR intervals per minute", `${d.rr_len.min} to ${d.rr_len.max}`, `median ${d.rr_len.median}; padded to ${d.rr_len.p99} (99th percentile)`],
      ]);
    }
    renderSplit(R);
    if (!s2) return;

    // architecture
    const hp = s2.best_hp || {};
    const layers = [["Input", `${R.dataset?.rr_len?.p99 || 87} RR values`, ""],
      ["Masking", "skips padding", ""]];
    for (let i = 0; i < (hp.num_lstm_layers || 1); i++)
      layers.push([`LSTM ${hp["lstm_units_" + i]}`, `${hp["lstm_activation_" + i]}, dropout ${hp["lstm_dropout_" + i]}`, "main"]);
    layers.push([`Dense ${hp.dense_units}`, hp.dense_activation, ""], ["Dense 1", "sigmoid: P(apnea)", ""]);
    $("arch").replaceChildren(...layers.map(([b, sp, c]) => el("div", { class: "layer " + c }, el("b", {}, b), el("span", {}, sp))),
      el("span", { class: "arch-total" }, `${int(s2.params)} parameters, learning rate ${fmt(hp.learning_rate, 5)}`));

    // tuner trials
    const tr = s2.tuner?.results || [];
    if (tr.length) {
      const desc = (h) => {
        const parts = [];
        for (let i = 0; i < h.num_lstm_layers; i++) parts.push(`LSTM ${h["lstm_units_" + i]} ${h["lstm_activation_" + i]}`);
        return parts.join(" + ") + `, dense ${h.dense_units} ${h.dense_activation}`;
      };
      $("tuner").replaceChildren(el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Trial (best first)"), el("th", {}, "Configuration"), el("th", { class: "n" }, "Validation AUC"))),
        el("tbody", {}, tr.map((t, i) => el("tr", { class: i === 0 ? "best" : null },
          el("td", {}, String(i + 1)), el("td", {}, desc(t.hp)), el("td", { class: "n" }, fmt(t.val_auc, 3)))))),
        el("p", { class: "caption" }, `${s2.tuner.trials} random-search trials, up to ${s2.tuner.epochs} epochs each, early stopping on validation AUC.`));
    }

    // results with threshold toggle
    const options = [["Threshold 0.50", s2.test]];
    if (s2.test_tuned) options.push([`Tuned threshold ${fmt(s2.threshold)}`, s2.test_tuned]);
    const seg = $("s2-threshold");
    const show = (k) => {
      const t = options[k][1];
      statList($("s2-metrics"), [
        ["ROC-AUC", fmt(t.auc, 3), "threshold-free; 0.5 is chance"],
        ["Balanced accuracy", pct(t.balanced_accuracy), "mean of the two recalls"],
        ["Apnea minutes caught", pct(t.apnea.recall, 0), "sensitivity", "apnea"],
        ["Normal minutes cleared", pct(t.normal.recall, 0), "specificity"],
        ["Apnea calls that were right", pct(t.apnea.precision, 0), "precision"],
        ["Accuracy", pct(t.accuracy), `${int(t.n)} test minutes`],
      ]);
      confusion($("s2-cm"), t.cm);
      $("s2-cm-caption").textContent = `LSTM confusion matrix on the six test patients, threshold ${fmt(t.threshold)}.`;
      [...seg.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === k)));
    };
    if (options.length > 1) {
      seg.hidden = false;
      seg.replaceChildren(...options.map(([label], i) => el("button", { type: "button", onclick: () => show(i) }, label)));
      show(1);
    } else show(0);

    $("s2-reading").replaceChildren(el("p", {},
      R.source === "pipeline"
        ? "AUC is the honest headline: it measures how well the model ranks apnea minutes above normal ones without committing to a threshold. The threshold then sets the trade-off. A screening tool wants high sensitivity, because a missed apnea costs a diagnosis and a false alarm costs a follow-up; a confirmatory tool wants the opposite."
        : "AUC of 0.76 on unseen patients is the honest headline: well above the 0.5 of chance, from heartbeat timing alone. The 81% sensitivity with 32% precision comes partly from the double imbalance correction described under Audit, which pushed the model toward calling apnea."));

    // curves
    if (s2.roc) {
      $("s2-curves").hidden = false;
      const series = [{ points: s2.roc.fpr.map((f, i) => [f, s2.roc.tpr[i]]) }];
      const legend = [{ text: `LSTM, AUC ${fmt(s2.test.auc, 3)}` }];
      if (R.stage1b?.roc) {
        series.push({ points: R.stage1b.roc.fpr.map((f, i) => [f, R.stage1b.roc.tpr[i]]), class: "ann" });
        legend.push({ text: `ANN, AUC ${fmt(R.stage1b.test.auc, 3)}`, class: "ann" });
      }
      lineChart($("roc"), {
        width: 460, height: 440, x: [0, 1], y: [0, 1], margin: { b: 44 },
        xTicks: [0, 0.25, 0.5, 0.75, 1], yTicks: [0, 0.25, 0.5, 0.75, 1],
        xTitle: "False alarm rate (1 - specificity)", yTitle: "Sensitivity",
        extras: [({ X, Y }) => s("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: "chance" })],
        series, legend: null, label: "ROC curves",
      });
      $("roc-caption").textContent = `ROC curves on the six test patients. Solid: ${legend[0].text}.` +
        (legend[1] ? ` Dashed: ${legend[1].text}, same split.` : "") + " Dotted diagonal: chance.";
      if (s2.history?.auc) historyChart($("s2-history"), s2.history, "auc", "AUC", [460, 440]);
    }

    // per-patient
    const pp = s2.per_patient;
    if (pp && pp.length) {
      $("patients-block").hidden = false;
      const maxIdx = Math.max(...pp.flatMap((p) => [p.true_index, p.pred_index]), 10);
      const dumb = (p) => {
        const w = 180, X = (v) => 6 + (v / maxIdx) * (w - 12);
        const svg = s("svg", { viewBox: `0 0 ${w} 18`, class: "dumbbell", role: "img",
          "aria-label": `true ${p.true_index}, predicted ${p.pred_index} apnea minutes per hour` });
        svg.append(s("line", { x1: 6, x2: w - 6, y1: 9, y2: 9, class: "track" }),
          s("line", { x1: X(p.true_index), x2: X(p.pred_index), y1: 9, y2: 9, class: "link" }),
          s("circle", { cx: X(p.true_index), cy: 9, r: 5, class: "true" }),
          s("circle", { cx: X(p.pred_index), cy: 9, r: 5, class: "pred" }));
        return svg;
      };
      const groups = { a: "apnea", b: "borderline", c: "control" };
      const ann = Object.fromEntries((R.stage1b?.per_patient || []).map((p) => [p.id, p]));
      $("patients").replaceChildren(el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Record"), el("th", {}, "Group"), el("th", { class: "n" }, "Minutes"),
          el("th", { class: "n" }, "LSTM AUC"), el("th", { class: "n" }, "ANN AUC"),
          el("th", { class: "n" }, "Apnea min/hour, experts"), el("th", { class: "n" }, "LSTM"),
          el("th", {}, "Experts (filled) vs LSTM (ring)"))),
        el("tbody", {}, pp.map((p) => el("tr", {},
          el("td", {}, el("b", {}, p.id)), el("td", { class: "group-" + p.group }, groups[p.group] || p.group),
          el("td", { class: "n" }, int(p.minutes)), el("td", { class: "n" }, p.auc == null ? "one class only" : fmt(p.auc, 3)),
          el("td", { class: "n" }, ann[p.id]?.auc == null ? "n/a" : fmt(ann[p.id].auc, 3)),
          el("td", { class: "n" }, fmt(p.true_index, 1)), el("td", { class: "n" }, fmt(p.pred_index, 1)),
          el("td", {}, dumb(p)))))),
        el("p", { class: "caption" }, `Scale 0 to ${Math.round(maxIdx)} apnea minutes per hour. AUC is undefined for a record whose minutes are all one class.`));
    }
  }

  function renderCompare(R) {
    const s1 = R.stage1, s1b = R.stage1b, s2 = R.stage2;
    const cols = [];
    if (s1) cols.push({ name: "Stage 1: ANN", sub: "one patient", patients: "1 (a01)", split: "Random over minutes", m: s1.test, params: s1.params });
    if (s1b) cols.push({ name: "Stage 1b: ANN", sub: "35 patients", patients: "35", split: "By patient", m: s1b.test_tuned || s1b.test, params: s1b.params });
    if (s2) cols.push({ name: "Stage 2: LSTM", sub: "35 patients", patients: "35", split: "By patient", m: s2.test_tuned || s2.test, params: s2.params });
    const rows = [
      ["Patients", (c) => c.patients],
      ["Split", (c) => c.split],
      ["Parameters", (c) => int(c.params)],
      ["Test minutes", (c) => int(c.m.n)],
      ["Normal minutes in test", (c) => int(c.m.normal.support)],
      ["ROC-AUC", (c) => fmt(c.m.auc, 3)],
      ["Accuracy", (c) => pct(c.m.accuracy)],
      ["Balanced accuracy", (c) => pct(c.m.balanced_accuracy)],
      ["Apnea minutes caught", (c) => pct(c.m.apnea.recall, 0)],
      ["Normal minutes cleared", (c) => pct(c.m.normal.recall, 0)],
      ["Question answered", (c) => c.split === "By patient" ? "Can it read a stranger?" : "Can it re-read a night it has seen?"],
    ];
    $("compare-table").replaceChildren(el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, ""), cols.map((c) => el("th", {}, c.name, el("br"), el("span", { style: "font-weight:400" }, c.sub))))),
      el("tbody", {}, rows.map(([k, f]) => el("tr", {}, el("td", {}, k), cols.map((c) => el("td", {}, f(c))))))));
    const note = $("compare-note");
    if (s1b && s2) {
      const better = s2.test.auc > s1b.test.auc;
      note.replaceChildren(el("p", {}, `Stage 1b is the fair comparison: the same seven features and the same network as Stage 1, but trained and tested on exactly the LSTM's patients. ${better
        ? `The LSTM's higher AUC (${fmt(s2.test.auc, 3)} against ${fmt(s1b.test.auc, 3)}) is the measured value of reading beats in order.`
        : `Here the feature-based ANN matches or beats the LSTM (${fmt(s1b.test.auc, 3)} against ${fmt(s2.test.auc, 3)}): with one input stream and 24 training patients, the ordering of beats adds less than expected. That is a real result, and worth stating.`} Where a column shows a tuned threshold, it was chosen on validation patients, never on the test set.`));
    } else {
      note.replaceChildren(el("p", {}, "After a pipeline run, a third column appears: the Stage 1 network trained on the same 35-patient split as the LSTM, which isolates what reading beats in order is actually worth."));
    }
  }

  // ------------------------------------------------------------------ demo
  function renderDemo(M) {
    const host = $("demo-body");
    if (!M || !M.lstm) {
      host.replaceChildren(el("div", { class: "empty" },
        el("h3", {}, "The live model switches on after the first training run"),
        el("p", {}, "This page runs the trained LSTM and ANN directly in the browser from docs/data/model.json. That file is written by the pipeline. Run it once, commit the two files in docs/data, and this panel becomes interactive with real minutes from the test patients."),
        el("pre", { class: "code" }, el("code", {}, "python scripts/run_pipeline.py\ngit add docs/data\ngit commit -m \"Add trained model\"\ngit push"))));
      return;
    }
    const API = window.ApneaModel;
    const samples = M.samples || [];
    let active = null;

    const list = el("ul", { class: "sample-list", "aria-label": "Test-patient minutes" });
    const ta = el("textarea", { id: "rr-input", spellcheck: "false", placeholder: "0.82, 0.85, 0.91, 0.88, ..." });
    const err = el("p", { class: "err", role: "alert" });
    const chart = el("div", { class: "chart" });
    const verdict = el("p", { class: "verdict" });
    const verdictSub = el("p", { class: "verdict-sub" });
    const meters = el("div", { class: "meters" });
    const file = el("input", { type: "file", accept: ".txt,.csv", class: "visually-hidden" });
    const fileBtn = el("label", { class: "btn" }, "Load a .txt or .csv", file);

    samples.forEach((smp, i) => {
      const b = el("button", { type: "button", "aria-pressed": "false", onclick: () => pick(i) },
        el("span", {}, `${smp.record}, minute ${smp.minute}`),
        el("span", { class: "tag " + (smp.label ? "apnea" : "normal") }, smp.label ? "labelled apnea" : "labelled normal"));
      list.append(el("li", {}, b));
    });

    function meter(name, p, thr, note) {
      return el("div", {},
        el("div", { class: "meter-head" }, el("b", {}, name), el("span", {}, `P(apnea) ${fmt(p, 3)}`)),
        el("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": "1", "aria-valuenow": p.toFixed(3), "aria-label": `${name} probability of apnea` },
          el("div", { class: "meter-fill", style: `width:${(p * 100).toFixed(1)}%` }),
          el("div", { class: "meter-thr", style: `left:calc(${(thr * 100).toFixed(1)}% - 1px)`, title: `threshold ${fmt(thr)}` })),
        el("p", { class: "hint" }, note));
    }

    function run(rr, truth, pyRef) {
      err.textContent = "";
      const clean = API.cleanRR(rr);
      if (clean.length < 10) {
        err.textContent = `Found ${clean.length} usable RR intervals between 0.3 and 2.0 seconds. Enter at least 10 (a typical minute has 50 to 90).`;
        return;
      }
      const pL = API.predictLSTM(M.lstm, clean), pA = API.predictANN(M.ann, clean);
      const tL = M.lstm.threshold, tA = M.ann.threshold;
      const isApnea = pL >= tL;
      verdict.className = "verdict " + (isApnea ? "apnea" : "normal");
      verdict.textContent = isApnea ? "LSTM: apnea minute" : "LSTM: normal minute";
      const bits = [`${clean.length} intervals, mean heart rate ${Math.round(60 / (clean.reduce((a, b) => a + b, 0) / clean.length))} bpm.`];
      if (rr.length !== clean.length) bits.push(`${rr.length - clean.length} values outside 0.3 to 2.0 s were dropped as artifacts.`);
      if (truth != null) bits.push(`Experts labelled this minute ${truth ? "apnea" : "normal"}; the LSTM is ${(isApnea === !!truth) ? "right" : "wrong"} here.`);
      verdictSub.textContent = bits.join(" ");
      const match = pyRef && Math.abs(pyRef.py_lstm - pL) < 1e-4 ? " Matches the Python model's output for this minute." : "";
      meters.replaceChildren(
        meter("LSTM, reads the beats in order", pL, tL, `Calls apnea above ${fmt(tL)} (threshold set on validation patients).${match}`),
        meter("ANN, seven summary statistics", pA, tA, `Calls apnea above ${fmt(tA)}. Same 35-patient training split.`));
      tachogram(chart, [{ rr: clean, label: truth }], { height: 240, width: 760, bandText: false, label: "RR intervals of the selected minute" });
    }

    function pick(i) {
      active = i;
      [...list.querySelectorAll("button")].forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
      const smp = samples[i];
      ta.value = smp.rr.join(", ");
      run(smp.rr, smp.label, smp);
    }
    function fromText() {
      const vals = (ta.value.match(/-?\d*\.?\d+(e-?\d+)?/gi) || []).map(Number);
      const ms = vals.length && vals.filter((v) => v > 0).every((v) => v > 20);   // looks like milliseconds
      [...list.querySelectorAll("button")].forEach((b) => b.setAttribute("aria-pressed", "false"));
      active = null;
      run(ms ? vals.map((v) => v / 1000) : vals, null, null);
    }
    file.addEventListener("change", async () => {
      if (!file.files[0]) return;
      ta.value = await file.files[0].text();
      fromText();
    });

    host.replaceChildren(el("div", { class: "demo-grid" },
      el("div", { class: "picker" },
        el("div", {}, el("label", {}, "Real minutes from the test patients"), list),
        el("div", {},
          el("label", { for: "rr-input" }, "Or paste RR intervals"),
          ta,
          el("p", { class: "hint" }, "Seconds or milliseconds, separated by commas, spaces or new lines. One minute is about 60 values."),
          el("p", { class: "hero-actions", style: "margin-top:.6rem" },
            el("button", { type: "button", class: "btn btn-primary", onclick: fromText }, "Classify this minute"), fileBtn),
          err)),
      el("div", { class: "readout", "aria-live": "polite" },
        el("div", {}, verdict, verdictSub), el("div", { class: "strip strip-chart" }, chart), meters)));
    if (samples.length) pick(samples.findIndex((x) => x.label === 1) >= 0 ? samples.findIndex((x) => x.label === 1) : 0);
  }

  // ------------------------------------------------------------------ boot
  function repoLink() {
    const m = location.hostname.match(/^([^.]+)\.github\.io$/i);
    if (!m) return;
    const repo = location.pathname.split("/").filter(Boolean)[0];
    const url = repo ? `https://github.com/${m[1]}/${repo}` : `https://github.com/${m[1]}`;
    const a = $("repo-link");
    a.href = url; a.hidden = false;
    $("clone-url").textContent = url;
  }

  async function boot() {
    repoLink();
    const [fetched, M] = await Promise.all([loadJSON("data/results.json"), loadJSON("data/model.json")]);
    let R = fetched;
    if (!R) { try { R = JSON.parse($("results-fallback").textContent); } catch (_) { R = {}; } }

    const heroRR = R.dataset?.sample_rr?.rr || R.stage1?.first_minute_rr || [0.89, 0.84, 0.83, 0.83, 0.88, 0.93, 0.98, 0.94];
    drawHero(heroRR);
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => drawHero(heroRR), 150); });

    const parts = [renderBanner, renderSignal, renderStage1, renderStage2, renderCompare];
    for (const f of parts) { try { f(R); } catch (e) { console.error(f.name, e); } }
    try { renderDemo(M); } catch (e) { console.error("demo", e); }
    if (R.generated) $("generated").textContent = `Results generated ${R.generated}${R.versions ? ` with TensorFlow ${R.versions.tensorflow}` : ""}.`;
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
