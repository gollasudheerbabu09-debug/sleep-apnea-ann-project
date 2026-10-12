"""Patient-independent splitting: no record ever appears on both sides."""
import numpy as np
from sklearn.model_selection import train_test_split

from .config import SEED


def patient_split(patient_ids, seed: int = SEED):
    """70/15/15 split over records. Reproduces the original notebook's split exactly."""
    unique = np.unique(patient_ids)
    train, temp = train_test_split(unique, test_size=0.30, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    return [str(p) for p in train], [str(p) for p in val], [str(p) for p in test]
