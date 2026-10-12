"""Scaling and padding RR series for the LSTM."""
import numpy as np

from .config import MASK_VALUE, MAX_LEN


class RRScaler:
    """StandardScaler over individual RR values, fit on training minutes only."""

    def fit(self, series):
        flat = np.concatenate(series).astype(np.float64)
        self.mean_ = float(flat.mean())
        self.scale_ = float(flat.std())
        return self

    def transform(self, series):
        return [(np.asarray(s, np.float64) - self.mean_) / self.scale_ for s in series]


def pad(series, max_len: int = MAX_LEN, value: float = MASK_VALUE) -> np.ndarray:
    """Post-pad / post-truncate to (n, max_len, 1), same as Keras pad_sequences."""
    out = np.full((len(series), max_len), value, dtype=np.float32)
    for i, s in enumerate(series):
        s = np.asarray(s)[:max_len]
        out[i, : len(s)] = s
    return out[..., np.newaxis]
