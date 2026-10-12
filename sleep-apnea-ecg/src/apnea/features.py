"""The seven summary statistics used by the ANN."""
import numpy as np
from scipy.stats import kurtosis, skew


def rr_features(rr: np.ndarray) -> np.ndarray:
    rr = np.asarray(rr, dtype=np.float64)
    return np.array([
        rr.max(), rr.min(), rr.mean(), np.median(rr), rr.std(),
        kurtosis(rr), skew(rr),          # scipy defaults: Fisher, biased
    ])


def feature_matrix(series) -> np.ndarray:
    return np.vstack([rr_features(s) for s in series])
