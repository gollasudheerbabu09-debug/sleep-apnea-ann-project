"""Every constant the pipeline uses, in one place."""

FS = 100                      # Apnea-ECG sampling frequency (Hz)
WINDOW_SEC = 60               # one label per minute
WINDOW = FS * WINDOW_SEC

RR_MIN, RR_MAX = 0.3, 2.0     # seconds; 30-200 bpm. Anything else is a detection artifact
MIN_RR_PER_MINUTE = 30        # drop minutes where beat detection clearly failed

MAX_LEN = 87                  # 99th percentile of RR-intervals-per-minute
MASK_VALUE = -10.0            # far outside the standardized RR range

SEED = 42

RECORDS = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)

FEATURE_NAMES = ["Maximum", "Minimum", "Mean", "Median", "Std. deviation", "Kurtosis", "Skewness"]
