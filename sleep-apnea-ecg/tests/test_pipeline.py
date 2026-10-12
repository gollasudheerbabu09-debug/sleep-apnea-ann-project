"""Fast tests: no TensorFlow and no dataset download needed."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apnea import config as C                       # noqa: E402
from apnea.data import build_rr_dataset, minute_rr, segment_record  # noqa: E402
from apnea.export import forward                      # noqa: E402
from apnea.features import rr_features                # noqa: E402
from apnea.sequences import RRScaler, pad             # noqa: E402
from apnea.splits import patient_split                # noqa: E402

# First minute of record a01, as printed by the original LSTM notebook (cell 11).
A01_MINUTE0 = [0.89, 0.84, 0.83, 0.83, 0.88, 0.93, 0.98, 0.94, 0.93, 0.93, 1.01, 1.02, 1.01, 0.99,
               0.97, 0.95, 0.98, 1.0, 0.95, 0.9, 0.85, 0.86, 0.97, 0.93, 0.93, 0.83, 0.78, 0.77,
               0.81, 0.81, 0.84, 0.84, 0.81, 0.82, 0.9, 1.0, 0.93, 0.88, 0.86, 0.85, 0.87, 0.89,
               0.91, 0.92, 0.88, 0.97, 1.0, 0.94, 0.86, 0.82, 0.83, 0.88, 0.88, 0.88, 0.92, 0.9,
               0.89, 0.93, 0.9, 0.9, 0.91, 0.91, 0.88, 0.88, 0.93, 0.89]


def test_split_matches_original_notebook():
    train, val, test = patient_split(np.array(C.RECORDS))
    assert sorted(test) == sorted(["c05", "c02", "a17", "a10", "b05", "a14"])
    assert sorted(val) == sorted(["a09", "a16", "a13", "b02", "a20"])
    assert len(train) == 24
    assert not set(train) & set(val) | set(train) & set(test) | set(val) & set(test)


def test_minute_rr_drops_artifacts_and_stays_inside_the_minute():
    # beats every 0.8 s, one spurious beat 0.1 s after another, one beat in the next minute
    peaks = np.array([0, 80, 160, 170, 250, 330, 6010])
    rr = minute_rr(peaks, 0)
    assert np.all((rr >= C.RR_MIN) & (rr <= C.RR_MAX))
    assert 0.1 not in rr
    assert len(rr) == 4          # 80, 80, 80, 80 samples kept; the 10-sample one dropped


def test_segments_keep_minute_and_label_aligned():
    fs = C.FS
    peaks = np.arange(0, 3 * 60 * fs, 90)   # 3 minutes of steady beats
    labels = np.array(["N", "A", "N"])
    out = list(segment_record(peaks, labels))
    assert [m for m, _, _ in out] == [0, 1, 2]
    assert [lab for _, _, lab in out] == [0, 1, 0]


def test_dataset_length_filter():
    def loader(_path, rec):
        # minute 2: only 15 valid RR intervals (detection dropout) -> removed by the length filter
        peaks = np.concatenate([np.arange(0, 6000, 90), np.arange(6000, 9000, 190)])
        return peaks, np.array(["A", "N"])

    ds = build_rr_dataset(Path("."), records=["x01"], loader=loader)
    assert ds.raw_count == 2 and len(ds.y) == 1 and ds.y[0] == 1 and ds.minute[0] == 0


def test_features_match_scipy_definitions():
    rr = np.array(A01_MINUTE0)
    f = rr_features(rr)
    np.testing.assert_allclose(f, [rr.max(), rr.min(), rr.mean(), np.median(rr), rr.std(),
                                   kurtosis(rr), skew(rr)])


def test_scaler_and_padding():
    s = RRScaler().fit([np.array([0.8, 1.0]), np.array([0.9])])
    assert abs(s.mean_ - 0.9) < 1e-12
    X = pad([np.ones(100), np.ones(3)])
    assert X.shape == (2, C.MAX_LEN, 1)
    assert (X[1, 3:, 0] == C.MASK_VALUE).all() and (X[0, :, 0] == 1).all()


def _random_lstm_spec(rng, units=5, dense=4, act="tanh"):
    return [
        {"type": "lstm", "units": units, "activation": act, "recurrent_activation": "sigmoid",
         "return_sequences": False, "kernel": rng.normal(size=(1, 4 * units)).tolist(),
         "recurrent_kernel": rng.normal(size=(units, 4 * units)).tolist(),
         "bias": rng.normal(size=4 * units).tolist()},
        {"type": "dense", "activation": "tanh", "kernel": rng.normal(size=(units, dense)).tolist(),
         "bias": rng.normal(size=dense).tolist()},
        {"type": "dense", "activation": "sigmoid", "kernel": rng.normal(size=(dense, 1)).tolist(),
         "bias": [0.1]},
    ]


def _reference_lstm(spec, seq):
    """Independent, gate-by-gate Keras LSTM written separately from export.forward."""
    L, D1, D2 = spec
    u = L["units"]
    W, U, b = np.array(L["kernel"]), np.array(L["recurrent_kernel"]), np.array(L["bias"])
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    gates = {g: (W[:, k * u:(k + 1) * u], U[:, k * u:(k + 1) * u], b[k * u:(k + 1) * u])
             for k, g in enumerate("ifco")}
    h = c = np.zeros(u)
    for x in seq:
        pre = {g: x * Wg[0] + h @ Ug + bg for g, (Wg, Ug, bg) in gates.items()}
        c = sig(pre["f"]) * c + sig(pre["i"]) * np.tanh(pre["c"])
        h = sig(pre["o"]) * np.tanh(c)
    d = np.tanh(h @ np.array(D1["kernel"]) + np.array(D1["bias"]))
    return float(sig(d @ np.array(D2["kernel"]) + np.array(D2["bias"]))[0])


def test_numpy_lstm_matches_reference():
    rng = np.random.default_rng(0)
    spec = _random_lstm_spec(rng)
    for _ in range(5):
        seq = rng.normal(size=rng.integers(30, 90))
        assert abs(forward(spec, seq) - _reference_lstm(spec, seq)) < 1e-10


def test_browser_lstm_matches_python(tmp_path):
    """The JavaScript inference used by the website must agree with Python."""
    import json
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node not installed")
    rng = np.random.default_rng(1)
    spec = _random_lstm_spec(rng, units=8, act="relu")
    seqs = [rng.normal(size=int(n)).round(4).tolist() for n in rng.integers(30, 110, size=6)]
    expected = [forward(spec, np.array(s)[: C.MAX_LEN]) for s in seqs]
    (tmp_path / "case.json").write_text(json.dumps({"layers": spec, "seqs": seqs}))
    js = f"""
      const m = require({json.dumps(str(ROOT / 'docs' / 'js' / 'model.js'))});
      const c = require({json.dumps(str(tmp_path / 'case.json'))});
      console.log(JSON.stringify(c.seqs.map(s => m.forward(c.layers, s.slice(0, {C.MAX_LEN})))));
    """
    got = json.loads(subprocess.check_output(["node", "-e", js], text=True))
    np.testing.assert_allclose(got, expected, atol=1e-9)
