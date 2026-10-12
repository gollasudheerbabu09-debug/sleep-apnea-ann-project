"""Run the whole project end to end and write everything the website shows.

    python scripts/run_pipeline.py              # full run (~15-30 min on a laptop CPU)
    python scripts/run_pipeline.py --quick      # smoke test in a few minutes

Outputs
    docs/data/results.json   every number and curve on the site
    docs/data/model.json     trained weights + sample minutes for the in-browser demo
    models/*.keras           the trained Keras models
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from apnea import config as C  # noqa: E402
from apnea.data import (build_rr_dataset, detect_rr_scipy, ecg_windows,  # noqa: E402
                        find_data_path, read_annotations, read_ecg)
from apnea.evaluate import (history_dict, metrics, per_patient, roc_points,  # noqa: E402
                            youden_threshold)
from apnea.export import export_layers, forward, max_abs_diff  # noqa: E402
from apnea.features import feature_matrix, rr_features  # noqa: E402
from apnea.sequences import RRScaler, pad  # noqa: E402
from apnea.splits import patient_split  # noqa: E402


def class_weights(y):
    from sklearn.utils.class_weight import compute_class_weight
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    return {0: float(w[0]), 1: float(w[1])}


def log(msg):
    print(f"\n=== {msg}", flush=True)


# --------------------------------------------------------------------------- Stage 1
def stage1(path, args):
    """Original experiment, bugs fixed: one patient (a01), own R-peak detection, random split."""
    import keras
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from apnea.models import build_ann

    log("Stage 1 - ANN on record a01 (single patient, random split)")
    windows = ecg_windows(read_ecg(path, "a01"))
    _, labels = read_annotations(path, "a01")
    n = min(len(windows), len(labels))            # 492 windows, 489 labels -> 489

    feats, ys, first_rr = [], [], None
    for i in range(n):                            # keep window i paired with label i
        rr = detect_rr_scipy(windows[i])
        if len(rr) < 2:
            continue
        if first_rr is None:
            first_rr = rr
        feats.append(rr_features(rr))
        ys.append(int(labels[i] == "A"))
    X, y = np.array(feats), np.array(ys)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=C.SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)           # fit on training rows only
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    keras.utils.set_random_seed(C.SEED)
    model = build_ann()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    hist = model.fit(X_tr, y_tr, epochs=args.ann_epochs, batch_size=32, validation_split=0.2,
                     callbacks=[es], verbose=0)
    prob = model.predict(X_te, verbose=0).ravel()
    model.save(ROOT / "models" / "ann_a01.keras")

    res = metrics(y_te, prob)
    res["baseline_accuracy"] = round(float(y_te.mean()), 4)   # "always say apnea"
    print(f"accuracy {res['accuracy']}  vs always-apnea baseline {res['baseline_accuracy']}  "
          f"normal recall {res['normal']['recall']}")
    return {
        "record": "a01",
        "windows": int(len(windows)),
        "labels": int(len(labels)),
        "segments": int(len(y)),
        "normal": int((y == 0).sum()),
        "apnea": int((y == 1).sum()),
        "params": int(model.count_params()),
        "test": res,
        "history": history_dict(hist),
        "first_minute_rr": [round(float(v), 3) for v in first_rr],
    }


# --------------------------------------------------------------------------- data
def load_dataset(path):
    log("Building the 35-record RR dataset from .qrs / .apn annotations")
    ds = build_rr_dataset(path)
    lengths = np.array(ds.raw_lengths)               # before the 30-interval filter
    train_p, val_p, test_p = patient_split(ds.patient)
    split = {"train": train_p, "val": val_p, "test": test_p, "counts": {}}
    for name, ids in (("train", train_p), ("val", val_p), ("test", test_p)):
        m = np.isin(ds.patient, ids)
        split["counts"][name] = {"records": len(ids), "n": int(m.sum()),
                                 "normal": int((ds.y[m] == 0).sum()), "apnea": int((ds.y[m] == 1).sum())}
    info = {
        "records": len(C.RECORDS),
        "segments_raw": int(ds.raw_count),
        "segments": int(len(ds.y)),
        "normal": int((ds.y == 0).sum()),
        "apnea": int((ds.y == 1).sum()),
        "min_rr_per_minute": C.MIN_RR_PER_MINUTE,
        "rr_len": {"min": int(lengths.min()), "max": int(lengths.max()),
                   "median": float(np.median(lengths)),
                   "p95": float(np.percentile(lengths, 95)), "p99": float(np.percentile(lengths, 99))},
        "per_record": ds.per_record,
    }
    # A real stretch of record a01 (minutes 4-27) spanning the normal -> apnea transition,
    # long enough to show the cyclic heart-rate pattern. Used by the website.
    stretch = []
    for m in range(4, 28):
        hit = np.flatnonzero((ds.patient == "a01") & (ds.minute == m))
        if len(hit):
            i = hit[0]
            stretch.append({"minute": m, "label": int(ds.y[i]), "rr": [round(float(v), 3) for v in ds.X[i]]})
    info["sample_stretch"] = {"record": "a01", "minutes": stretch}
    first = np.flatnonzero((ds.patient == "a01") & (ds.minute == 0))
    if len(first):
        info["sample_rr"] = {"record": "a01", "minute": 0, "label": int(ds.y[first[0]]),
                             "rr": [round(float(v), 3) for v in ds.X[first[0]]]}
    print(f"{info['segments']} minutes ({info['normal']} normal / {info['apnea']} apnea); "
          f"test records {test_p}")
    masks = {k: np.isin(ds.patient, split[k]) for k in ("train", "val", "test")}
    return ds, info, split, masks


# --------------------------------------------------------------------------- Stage 1b
def stage1b(ds, masks, args):
    """Same ANN + 7 features, but on all 35 records with the patient split: the fair comparison."""
    import keras
    from sklearn.preprocessing import StandardScaler
    from apnea.models import build_ann, callbacks

    log("Stage 1b - same ANN, all 35 records, patient-independent split")
    F = feature_matrix(ds.X)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    scaler = StandardScaler().fit(F[tr])
    Xs = scaler.transform(F)

    keras.utils.set_random_seed(C.SEED)
    model = build_ann()
    hist = model.fit(Xs[tr], ds.y[tr], validation_data=(Xs[va], ds.y[va]), epochs=args.ann_epochs,
                     batch_size=64, class_weight=class_weights(ds.y[tr]),
                     callbacks=callbacks(patience=15), verbose=0)
    p_val = model.predict(Xs[va], verbose=0).ravel()
    p_te = model.predict(Xs[te], verbose=0).ravel()
    thr = youden_threshold(ds.y[va], p_val)
    model.save(ROOT / "models" / "ann_35_patients.keras")

    layers = export_layers(model)
    diff = max_abs_diff(layers, Xs[te], p_te)
    assert diff < 1e-4, f"ANN export mismatch {diff}"

    out = {
        "params": int(model.count_params()),
        "threshold": round(thr, 4),
        "val_auc": metrics(ds.y[va], p_val)["auc"],
        "test": metrics(ds.y[te], p_te, 0.5),
        "test_tuned": metrics(ds.y[te], p_te, thr),
        "roc": roc_points(ds.y[te], p_te),
        "per_patient": per_patient(ds.y[te], p_te, ds.patient[te], thr),
        "history": history_dict(hist),
        "export_max_abs_diff": diff,
    }
    print(f"test AUC {out['test']['auc']}")
    spec = {"layers": layers, "feature_mean": scaler.mean_.round(7).tolist(),
            "feature_scale": scaler.scale_.round(7).tolist(), "threshold": round(thr, 4)}
    return out, spec, p_te


# --------------------------------------------------------------------------- Stage 2
def stage2(ds, masks, args):
    import keras
    import keras_tuner as kt
    from apnea.models import active_hp, build_lstm, callbacks, lstm_hypermodel

    log("Stage 2 - LSTM on raw RR sequences, patient-independent split")
    tr, va, te = masks["train"], masks["val"], masks["test"]
    scaler = RRScaler().fit([ds.X[i] for i in np.flatnonzero(tr)])
    scaled = scaler.transform(ds.X)
    X = pad(scaled)
    cw = class_weights(ds.y[tr])                  # weights only - no undersampling on top

    keras.utils.set_random_seed(C.SEED)
    tuner = kt.RandomSearch(lstm_hypermodel, objective=kt.Objective("val_auc", "max"),
                            max_trials=args.trials, seed=C.SEED, overwrite=True,
                            directory=str(ROOT / ".tuner"), project_name="lstm")
    t0 = time.time()
    tuner.search(X[tr], ds.y[tr], validation_data=(X[va], ds.y[va]), epochs=args.tune_epochs,
                 batch_size=128, class_weight=cw, callbacks=callbacks(patience=4), verbose=1)
    trials = []
    for t in tuner.oracle.get_best_trials(num_trials=args.trials):
        trials.append({"val_auc": round(float(t.score), 4), "hp": active_hp(t.hyperparameters.values)})
    best_hp = active_hp(tuner.get_best_hyperparameters(1)[0].values)
    print("best hyperparameters:", best_hp)

    # Final model: fresh build from the best configuration, trained longer. The same
    # object is trained, evaluated and exported (the original evaluated a different one).
    keras.utils.set_random_seed(C.SEED)
    model = build_lstm(best_hp)
    hist = model.fit(X[tr], ds.y[tr], validation_data=(X[va], ds.y[va]), epochs=args.epochs,
                     batch_size=128, class_weight=cw, callbacks=callbacks(patience=8), verbose=2)
    p_val = model.predict(X[va], verbose=0).ravel()
    p_te = model.predict(X[te], verbose=0).ravel()
    thr = youden_threshold(ds.y[va], p_val)
    model.save(ROOT / "models" / "lstm_35_patients.keras")

    layers = export_layers(model)
    te_idx = np.flatnonzero(te)
    check = te_idx[:: max(1, len(te_idx) // 200)]           # 200 minutes is plenty to verify
    diff = max_abs_diff(layers, [scaled[i][: C.MAX_LEN] for i in check],
                        model.predict(X[check], verbose=0))
    assert diff < 1e-4, f"LSTM export mismatch {diff}"

    out = {
        "tuner": {"trials": args.trials, "epochs": args.tune_epochs,
                  "seconds": round(time.time() - t0), "results": trials},
        "best_hp": best_hp,
        "params": int(model.count_params()),
        "threshold": round(thr, 4),
        "val_auc": metrics(ds.y[va], p_val)["auc"],
        "test": metrics(ds.y[te], p_te, 0.5),
        "test_tuned": metrics(ds.y[te], p_te, thr),
        "roc": roc_points(ds.y[te], p_te),
        "per_patient": per_patient(ds.y[te], p_te, ds.patient[te], thr),
        "history": history_dict(hist),
        "export_max_abs_diff": diff,
    }
    print(f"test AUC {out['test']['auc']}  (threshold {thr:.3f} chosen on validation)")
    spec = {"layers": layers, "rr_mean": scaler.mean_, "rr_scale": scaler.scale_,
            "max_len": C.MAX_LEN, "threshold": round(thr, 4)}
    return out, spec, p_te


# --------------------------------------------------------------------------- export
def demo_samples(ds, masks, p_lstm, p_ann, per_record=4):
    """A few real test-patient minutes (both classes) for the website's demo picker."""
    rng = np.random.default_rng(C.SEED)
    te_idx = np.flatnonzero(masks["test"])
    pos = {i: k for k, i in enumerate(te_idx)}
    samples = []
    for pid in sorted(np.unique(ds.patient[te_idx])):
        for label in (0, 1):
            cand = [i for i in te_idx if ds.patient[i] == pid and ds.y[i] == label]
            for i in rng.choice(cand, size=min(per_record // 2, len(cand)), replace=False):
                samples.append({
                    "record": str(pid), "minute": int(ds.minute[i]), "label": int(label),
                    "rr": [round(float(v), 3) for v in ds.X[i]],
                    "py_lstm": round(float(p_lstm[pos[i]]), 6), "py_ann": round(float(p_ann[pos[i]]), 6),
                })
    return samples


def embed_fallback(results):
    """Keep the copy embedded in index.html current, so the page also works opened from disk."""
    import re
    page = ROOT / "docs" / "index.html"
    html = page.read_text()
    block = json.dumps(results, separators=(",", ":")).replace("</", "<\\/")
    html, n = re.subn(r'(<script id="results-fallback" type="application/json">).*?(</script>)',
                      lambda m: m.group(1) + block + m.group(2), html, flags=re.S)
    if n:
        page.write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10, help="Keras Tuner trials (original: 4)")
    ap.add_argument("--tune-epochs", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=60, help="final LSTM training epochs (early stopping)")
    ap.add_argument("--ann-epochs", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--data", default=None, help="folder with the Apnea-ECG files")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.tune_epochs, args.epochs, args.ann_epochs = 2, 2, 3, 20

    import os
    if args.data:
        os.environ["APNEA_DATA"] = args.data
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "docs" / "data").mkdir(parents=True, exist_ok=True)

    path = find_data_path(ROOT / "data" / "apnea-ecg")
    print("data:", path)

    s1 = stage1(path, args)
    ds, info, split, masks = load_dataset(path)
    s1b, ann_spec, p_ann = stage1b(ds, masks, args)
    s2, lstm_spec, p_lstm = stage2(ds, masks, args)

    import keras
    import tensorflow as tf
    results = {
        "source": "pipeline",
        "quick": bool(args.quick),
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__},
        "dataset": info,
        "split": split,
        "stage1": s1,
        "stage1b": s1b,
        "stage2": s2,
    }
    (ROOT / "docs" / "data" / "results.json").write_text(json.dumps(results, indent=1))
    embed_fallback(results)

    model_json = {
        "generated": results["generated"],
        "lstm": lstm_spec,
        "ann": ann_spec,
        "samples": demo_samples(ds, masks, p_lstm, p_ann),
    }
    (ROOT / "docs" / "data" / "model.json").write_text(json.dumps(model_json, separators=(",", ":")))

    log("Done")
    print(f"Stage 1  (a01, random split)   accuracy {s1['test']['accuracy']}  AUC {s1['test']['auc']}")
    print(f"Stage 1b (ANN, unseen records) AUC {s1b['test']['auc']}")
    print(f"Stage 2  (LSTM, unseen records) AUC {s2['test']['auc']}")
    print("Wrote docs/data/results.json and docs/data/model.json - commit them to update the site.")


if __name__ == "__main__":
    main()
