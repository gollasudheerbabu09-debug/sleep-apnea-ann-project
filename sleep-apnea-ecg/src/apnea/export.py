"""Export trained Keras models to plain JSON, and a NumPy forward pass that mirrors them.

The website re-implements exactly this forward pass in JavaScript, so the trained model
runs in the visitor's browser with no server. `verify_*` checks the NumPy version against
Keras on real data before anything is written.
"""
from __future__ import annotations

import numpy as np

ACT = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "linear": lambda x: x,
}


def _act_name(fn) -> str:
    name = getattr(fn, "__name__", str(fn))
    if name not in ACT:
        raise ValueError(f"Unsupported activation for export: {name}")
    return name


def _round(a, d=7):
    return np.round(np.asarray(a, dtype=np.float64), d).tolist()


def export_layers(model) -> list[dict]:
    """Serialise LSTM and Dense layers; Masking/Dropout/Input carry no weights at inference."""
    out = []
    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls == "LSTM":
            k, rk, b = layer.get_weights()
            out.append({"type": "lstm", "units": int(layer.units),
                        "activation": _act_name(layer.activation),
                        "recurrent_activation": _act_name(layer.recurrent_activation),
                        "return_sequences": bool(layer.return_sequences),
                        "kernel": _round(k), "recurrent_kernel": _round(rk), "bias": _round(b)})
        elif cls == "Dense":
            k, b = layer.get_weights()
            out.append({"type": "dense", "activation": _act_name(layer.activation),
                        "kernel": _round(k), "bias": _round(b)})
        elif cls not in ("Masking", "Dropout", "InputLayer"):
            raise ValueError(f"Unsupported layer for export: {cls}")
    return out


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(layers: list[dict], x: np.ndarray) -> float:
    """One example. For LSTMs, `x` is the *unpadded* scaled sequence (masking == stopping early)."""
    h = np.asarray(x, dtype=np.float64)
    if layers and layers[0]["type"] == "lstm":
        h = h.reshape(-1, 1)
    for L in layers:
        if L["type"] == "lstm":
            W, U, b = (np.asarray(L[k]) for k in ("kernel", "recurrent_kernel", "bias"))
            u = L["units"]
            act, ract = ACT[L["activation"]], ACT[L["recurrent_activation"]]
            ht, ct, seq = np.zeros(u), np.zeros(u), []
            for xt in h:
                z = xt @ W + ht @ U + b                      # Keras gate order: i, f, c, o
                i, f, g, o = z[:u], z[u:2 * u], z[2 * u:3 * u], z[3 * u:]
                ct = ract(f) * ct + ract(i) * act(g)
                ht = ract(o) * act(ct)
                seq.append(ht)
            h = np.array(seq) if L["return_sequences"] else ht
        else:
            h = ACT[L["activation"]](h @ np.asarray(L["kernel"]) + np.asarray(L["bias"]))
    return float(np.ravel(h)[0])


def max_abs_diff(layers, inputs, keras_probs) -> float:
    ours = np.array([forward(layers, x) for x in inputs])
    return float(np.max(np.abs(ours - np.ravel(keras_probs))))
