"""Metrics that stay honest under class imbalance and across patients."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)


def _r(x, d=4):
    return None if x is None else round(float(x), d)


def metrics(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=np.float64).ravel()
    pred = (prob >= threshold).astype(int)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else None
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y)),
        "threshold": _r(threshold),
        "accuracy": _r((tp + tn) / len(y)),
        "balanced_accuracy": _r((r[0] + r[1]) / 2),
        "auc": _r(auc),
        "cm": cm.tolist(),
        "normal": {"precision": _r(p[0]), "recall": _r(r[0]), "f1": _r(f[0]), "support": int(s[0])},
        "apnea": {"precision": _r(p[1]), "recall": _r(r[1]), "f1": _r(f[1]), "support": int(s[1])},
    }


def youden_threshold(y, prob) -> float:
    """Threshold maximising sensitivity + specificity - 1. Chosen on validation data only."""
    fpr, tpr, thr = roc_curve(y, prob)
    best = np.argmax(tpr - fpr)
    return float(np.clip(thr[best], 1e-6, 1 - 1e-6))


def roc_points(y, prob, n: int = 80) -> dict:
    fpr, tpr, _ = roc_curve(y, prob)
    grid = np.linspace(0, 1, n)
    return {"fpr": [_r(v, 3) for v in grid], "tpr": [_r(v, 3) for v in np.interp(grid, fpr, tpr)]}


def per_patient(y, prob, patients, threshold: float) -> list[dict]:
    """Per-record performance plus a whole-night roll-up (apnea minutes per hour)."""
    y, prob, patients = np.asarray(y), np.asarray(prob).ravel(), np.asarray(patients)
    out = []
    for pid in sorted(np.unique(patients)):
        m = patients == pid
        yt, pt = y[m], prob[m]
        pred = pt >= threshold
        hours = m.sum() / 60
        out.append({
            "id": str(pid),
            "group": str(pid)[0],                    # a = apnea, b = borderline, c = control
            "minutes": int(m.sum()),
            "true_apnea_min": int(yt.sum()),
            "pred_apnea_min": int(pred.sum()),
            "true_index": _r(yt.sum() / hours, 1),   # apnea minutes per hour of recording
            "pred_index": _r(pred.sum() / hours, 1),
            "accuracy": _r((pred == yt).mean()),
            "auc": _r(roc_auc_score(yt, pt)) if len(np.unique(yt)) == 2 else None,
        })
    return out


def history_dict(history, keys=("loss", "val_loss", "auc", "val_auc", "accuracy", "val_accuracy")):
    h = history.history
    return {k: [_r(v) for v in h[k]] for k in keys if k in h}
