"""Keras model builders. TensorFlow is imported lazily so the rest of the package works without it."""
from __future__ import annotations

from .config import MASK_VALUE, MAX_LEN


def _keras():
    import keras
    return keras


def _metrics(keras):
    return [
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def build_ann(n_features: int = 7, lr: float = 1e-3):
    """Stage 1 network: 7 -> 40 -> 20 -> 1, unchanged from the original (1,161 parameters)."""
    keras = _keras()
    from keras import layers

    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(40, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(20, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="ann")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=_metrics(keras))
    return model


def build_lstm(hp: dict, max_len: int = MAX_LEN, mask_value: float = MASK_VALUE):
    """LSTM over a masked, padded RR sequence. `hp` is a plain dict of hyperparameters."""
    keras = _keras()
    from keras import layers

    model = keras.Sequential(name="lstm")
    model.add(keras.Input(shape=(max_len, 1)))
    model.add(layers.Masking(mask_value=mask_value))
    n = hp["num_lstm_layers"]
    for i in range(n):
        act = hp[f"lstm_activation_{i}"]
        model.add(layers.LSTM(
            hp[f"lstm_units_{i}"],
            activation=act,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            return_sequences=i < n - 1,
        ))
        model.add(layers.Dropout(hp[f"lstm_dropout_{i}"]))
    dact = hp["dense_activation"]
    model.add(layers.Dense(hp["dense_units"], activation=dact,
                           kernel_initializer="he_normal" if dact == "relu" else "glorot_uniform"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(hp["learning_rate"]),
                  loss="binary_crossentropy", metrics=_metrics(keras))
    return model


def lstm_hypermodel(hp):
    """Keras Tuner search space (same 7 dimensions as the original notebook)."""
    values = {"num_lstm_layers": hp.Int("num_lstm_layers", 1, 3)}
    for i in range(values["num_lstm_layers"]):
        values[f"lstm_units_{i}"] = hp.Int(f"lstm_units_{i}", 32, 128, step=32)
        values[f"lstm_activation_{i}"] = hp.Choice(f"lstm_activation_{i}", ["tanh", "relu"])
        values[f"lstm_dropout_{i}"] = hp.Float(f"lstm_dropout_{i}", 0.1, 0.5, step=0.1)
    values["dense_units"] = hp.Int("dense_units", 8, 64, step=8)
    values["dense_activation"] = hp.Choice("dense_activation", ["relu", "tanh"])
    values["learning_rate"] = hp.Float("learning_rate", 1e-4, 3e-3, sampling="log")
    return build_lstm(values)


def active_hp(values: dict) -> dict:
    """Drop the stale per-layer entries Keras Tuner keeps for layers that aren't built."""
    n = values["num_lstm_layers"]
    keep = {"num_lstm_layers", "dense_units", "dense_activation", "learning_rate"}
    keep |= {f"{k}_{i}" for i in range(n) for k in ("lstm_units", "lstm_activation", "lstm_dropout")}
    return {k: v for k, v in values.items() if k in keep}


def callbacks(patience: int = 8):
    keras = _keras()
    return [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                          patience=max(2, patience // 3), min_lr=1e-6),
    ]
