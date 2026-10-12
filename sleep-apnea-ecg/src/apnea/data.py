"""Locating, downloading and segmenting the PhysioNet Apnea-ECG database."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import FS, MIN_RR_PER_MINUTE, RECORDS, RR_MAX, RR_MIN, WINDOW

_KAGGLE_CANDIDATES = [
    "/kaggle/input/apneaecg/apnea-ecg",
    "/kaggle/input/apnea-ecg",
    "/kaggle/input/apnea-ecg-database-1.0.0",
]


def _has_records(path: Path) -> bool:
    return (path / "a01.qrs").exists() and (path / "a01.apn").exists()


def find_data_path(download_to: str | os.PathLike = "data/apnea-ecg") -> Path:
    """Return a folder holding the Apnea-ECG files, downloading them if needed.

    Search order: $APNEA_DATA, Kaggle input folders, a previous download,
    the kagglehub dataset ``ecerulm/apneaecg``, then PhysioNet via ``wfdb``.
    """
    candidates = [os.environ.get("APNEA_DATA"), *_KAGGLE_CANDIDATES, str(download_to)]
    for c in candidates:
        if c and _has_records(Path(c)):
            return Path(c)

    try:  # kagglehub (used by the original notebook)
        import kagglehub

        root = Path(kagglehub.dataset_download("ecerulm/apneaecg"))
        for p in [root / "apnea-ecg", root]:
            if _has_records(p):
                return p
    except Exception as exc:  # noqa: BLE001 - fall through to PhysioNet
        print(f"kagglehub unavailable ({exc}); downloading from PhysioNet instead")

    import wfdb

    dest = Path(download_to)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Apnea-ECG from PhysioNet into {dest} (~200 MB)...")
    wfdb.dl_database("apnea-ecg", str(dest), records=list(RECORDS))
    if not _has_records(dest):
        raise FileNotFoundError(f"Download finished but {dest}/a01.qrs is missing")
    return dest


def read_annotations(path: Path, record: str) -> tuple[np.ndarray, np.ndarray]:
    """R-peak sample indices (.qrs) and per-minute labels (.apn) for one record."""
    import wfdb

    qrs = wfdb.rdann(str(path / record), "qrs")
    apn = wfdb.rdann(str(path / record), "apn")
    return np.asarray(qrs.sample), np.asarray(apn.symbol)


def read_ecg(path: Path, record: str) -> np.ndarray:
    import wfdb

    rec = wfdb.rdrecord(str(path / record))
    return rec.p_signal[:, 0].astype(np.float32)


def minute_rr(r_peaks: np.ndarray, minute: int, fs: int = FS) -> np.ndarray:
    """RR intervals (s) from beats inside one minute, with artifacts removed."""
    start, end = minute * 60 * fs, (minute + 1) * 60 * fs
    beats = r_peaks[(r_peaks >= start) & (r_peaks < end)]
    rr = np.diff(beats) / fs
    return rr[(rr >= RR_MIN) & (rr <= RR_MAX)]


def segment_record(r_peaks: np.ndarray, labels: np.ndarray, fs: int = FS):
    """Yield (minute, rr, label) for every minute with a usable RR series."""
    for minute, sym in enumerate(labels):
        rr = minute_rr(r_peaks, minute, fs)
        if len(rr) < 2:
            continue
        yield minute, rr, int(sym == "A")


@dataclass
class RRDataset:
    X: list                                   # variable-length RR series, seconds
    y: np.ndarray                             # 1 = apnea minute
    patient: np.ndarray                       # record id per minute
    minute: np.ndarray                        # minute index within its record
    raw_count: int = 0                        # minutes before the length filter
    per_record: list = field(default_factory=list)
    raw_lengths: list = field(default_factory=list)   # RR count per minute before the filter

    def subset(self, mask: np.ndarray) -> "RRDataset":
        idx = np.flatnonzero(mask)
        return RRDataset([self.X[i] for i in idx], self.y[idx], self.patient[idx], self.minute[idx])


def build_rr_dataset(path: Path, records=RECORDS, min_rr: int = MIN_RR_PER_MINUTE,
                     loader=read_annotations) -> RRDataset:
    """All usable minutes from all records, using the database's own R-peak annotations."""
    X, y, pid, mins = [], [], [], []
    raw = 0
    per_record, raw_lengths = [], []
    for rec in records:
        r_peaks, labels = loader(path, rec)
        n_rec = n_apnea = n_raw = 0
        for minute, rr, label in segment_record(r_peaks, labels):
            raw += 1
            n_raw += 1
            raw_lengths.append(len(rr))
            if len(rr) < min_rr:
                continue
            X.append(rr.astype(np.float32))
            y.append(label)
            pid.append(rec)
            mins.append(minute)
            n_rec += 1
            n_apnea += label
        per_record.append({"id": rec, "n_raw": n_raw, "n": n_rec, "apnea": n_apnea,
                           "labels": int(len(labels))})
    return RRDataset(X, np.array(y, dtype=np.int64), np.array(pid), np.array(mins, dtype=np.int64),
                     raw_count=raw, per_record=per_record, raw_lengths=raw_lengths)


# --- Stage 1: the original single-patient pipeline (raw ECG -> own peak detection) ---

def ecg_windows(signal: np.ndarray, window: int = WINDOW) -> np.ndarray:
    n = len(signal) // window
    return signal[: n * window].reshape(n, window)


def detect_rr_scipy(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """R-peak detection exactly as in the original notebook."""
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(window, distance=60, prominence=0.5)
    return np.diff(peaks) / fs
