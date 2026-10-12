/* In-browser inference for the exported Keras models.
   Mirrors src/apnea/export.py `forward` exactly; tests/test_pipeline.py checks they agree. */
(function (root) {
  const ACT = {
    tanh: Math.tanh,
    relu: (x) => (x > 0 ? x : 0),
    sigmoid: (x) => 1 / (1 + Math.exp(-x)),
    linear: (x) => x,
  };

  function dense(L, h) {
    const W = L.kernel, b = L.bias, f = ACT[L.activation];
    const out = new Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < h.length; i++) s += h[i] * W[i][j];
      out[j] = f(s);
    }
    return out;
  }

  // Keras LSTM, gate order i, f, c, o. `seq` is an array of input vectors.
  function lstm(L, seq) {
    const u = L.units, W = L.kernel, U = L.recurrent_kernel, b = L.bias;
    const act = ACT[L.activation], ract = ACT[L.recurrent_activation];
    let h = new Array(u).fill(0), c = new Array(u).fill(0);
    const outs = [];
    const z = new Array(4 * u);
    for (const x of seq) {
      for (let k = 0; k < 4 * u; k++) {
        let s = b[k];
        for (let i = 0; i < x.length; i++) s += x[i] * W[i][k];
        for (let i = 0; i < u; i++) s += h[i] * U[i][k];
        z[k] = s;
      }
      const nh = new Array(u), nc = new Array(u);
      for (let j = 0; j < u; j++) {
        nc[j] = ract(z[u + j]) * c[j] + ract(z[j]) * act(z[2 * u + j]);
        nh[j] = ract(z[3 * u + j]) * act(nc[j]);
      }
      h = nh; c = nc;
      outs.push(h);
    }
    return L.return_sequences ? outs : h;
  }

  function forward(layers, x) {
    let h = layers.length && layers[0].type === "lstm" ? x.map((v) => [v]) : x.slice();
    for (const L of layers) h = L.type === "lstm" ? lstm(L, h) : dense(L, h);
    return h[0];
  }

  // ---- the seven RR statistics (numpy / scipy defaults: ddof=0, Fisher kurtosis, biased) ----
  function rrFeatures(rr) {
    const n = rr.length;
    const mean = rr.reduce((a, b) => a + b, 0) / n;
    let m2 = 0, m3 = 0, m4 = 0;
    for (const v of rr) { const d = v - mean; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
    m2 /= n; m3 /= n; m4 /= n;
    const s = [...rr].sort((a, b) => a - b);
    const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
    return [Math.max(...rr), Math.min(...rr), mean, median, Math.sqrt(m2),
            m2 > 0 ? m4 / (m2 * m2) - 3 : 0, m2 > 0 ? m3 / Math.pow(m2, 1.5) : 0];
  }

  function predictLSTM(spec, rr) {
    const x = rr.slice(0, spec.max_len).map((v) => (v - spec.rr_mean) / spec.rr_scale);
    return forward(spec.layers, x);
  }

  function predictANN(spec, rr) {
    const f = rrFeatures(rr).map((v, i) => (v - spec.feature_mean[i]) / spec.feature_scale[i]);
    return forward(spec.layers, f);
  }

  // Same artifact rule as the training pipeline.
  function cleanRR(values) {
    return values.filter((v) => Number.isFinite(v) && v >= 0.3 && v <= 2.0);
  }

  const api = { forward, rrFeatures, predictLSTM, predictANN, cleanRR };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ApneaModel = api;
})(this);
