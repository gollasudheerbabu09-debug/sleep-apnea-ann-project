[README.md](https://github.com/user-attachments/files/31774504/README.md)
# Sleep Apnea Detection from ECG — ANN → LSTM

Detecting obstructive sleep apnea from single-lead ECG, built in two stages: first a **feature-based Artificial Neural Network** on a single patient, then a **sequence-based LSTM** trained across 35 patients with a patient-independent evaluation protocol.

The second stage exists because of what the first stage revealed. That progression — and what it exposed about evaluating medical ML honestly — is the core of this project.

---

## Background

Sleep apnea is diagnosed by polysomnography, an overnight lab study that is expensive, uncomfortable, and capacity-limited. ECG offers a cheaper screening route: during an apnea event, the heart's autonomic response produces **cyclic variation in heart rate** — bradycardia during the apnea, tachycardia on arousal. That pattern lives in the **RR interval series** (the time between consecutive R-peaks), which is what both models in this project learn from.

Each minute of recording is labeled `A` (apnea) or `N` (normal), making this a **per-minute binary classification** problem.

---

## Dataset

**PhysioNet Apnea-ECG Database**

| Property | Value |
|---|---|
| Recordings used | 35 (`a01`–`a20`, `b01`–`b05`, `c01`–`c10`) |
| Sampling frequency | 100 Hz |
| Annotation granularity | 1 label per 60-second segment |
| Signal length (example, `a01`) | 2,957,000 samples (~8.2 hours) |
| Classes | `N` = normal, `A` = apnea |

- **Stage 1 (ANN)** used a single-patient CSV export of record `a01` with its `.txt` annotation file.
- **Stage 2 (LSTM)** used the full WFDB database via `wfdb`, loaded through `kagglehub`, reading the pre-computed `.qrs` R-peak annotations and `.apn` apnea labels for all 35 records.

---

## Repository Structure

```
.
├── sleepapenaannproject.ipynb    # Stage 1 — ANN on engineered RR features (single patient)
├── LSTM_sleep_apnea 1.ipynb      # Stage 2 — LSTM on raw RR sequences (35 patients)
├── SleepApnea-predictor.keras    # Saved ANN model
└── README.md
```

---

# Stage 1 — Artificial Neural Network

The first approach: turn each minute of ECG into a handful of summary statistics and classify those with a small dense network.

### Signal processing

The raw ECG is segmented into non-overlapping **60-second windows** (6,000 samples at 100 Hz), producing 492 complete windows from record `a01`.

R-peaks are detected in each window with `scipy.signal.find_peaks`:

```python
peaks, properties = find_peaks(window, distance=60, prominence=0.5)
```

- `distance=60` enforces a minimum 0.6 s gap between peaks — a physiological ceiling of ~100 bpm that stops the detector firing twice on one beat
- `prominence=0.5` selects the tall R spike and ignores the smaller P and T waves

RR intervals are then the differences between consecutive peaks, converted to seconds by dividing by `fs`.

### Feature engineering

Each variable-length RR series is collapsed into **7 statistical descriptors**:

```python
features = [
    np.max(rr), np.min(rr), np.mean(rr), np.median(rr),
    np.std(rr), kurtosis(rr), skew(rr)
]
```

Mean and median capture baseline heart rate; standard deviation and range capture heart rate variability; **kurtosis and skewness** capture the *shape* of the distribution, which is where the cyclic bradycardia-tachycardia signature of apnea shows up.

Result: a clean `(492, 7)` feature matrix. After aligning with the 488 available annotations, `(488, 7)`.

### Architecture

```python
model = Sequential()
model.add(Dense(40, activation='relu', input_shape=(7,)))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
```

**1,161 trainable parameters** — deliberately tiny, matched to a 7-dimensional input and a few hundred samples. Adam optimizer, binary cross-entropy, `StandardScaler` on the features, stratified 80/20 split, up to 200 epochs with `EarlyStopping(monitor='val_loss', patience=25)`.

### Results

| Metric | Value |
|---|---|
| Test Accuracy | **0.9796** |
| ROC-AUC | 1.00 |

```
              precision    recall  f1-score   support

  Normal (0)       1.00      0.50      0.67         4
   Apnea (1)       0.98      1.00      0.99        94

    accuracy                           0.98        98
   macro avg       0.99      0.75      0.83        98
weighted avg       0.98      0.98      0.98        98
```

Confusion matrix:

```
[[ 2   2]      2 of 4 normal minutes caught
 [ 0  94]]     94 of 94 apnea minutes caught
```

### Why 98% is not the real story

Record `a01` is a severe apnea patient: **470 apnea minutes against 18 normal minutes**. A model that predicts "apnea" for every single input scores ~96% on that distribution. The 97.96% here is only two points above that trivial baseline, and the confusion matrix shows why — the network catches **2 of 4** normal minutes in the test set.

Three problems compound:

1. **Single patient.** The model may have learned one person's cardiac signature, not apnea.
2. **Severe class imbalance.** Accuracy is close to meaningless at a 96/4 split.
3. **Random split.** With minutes from the same recording in both train and test, the split leaks patient-specific information across the boundary.

A test set of 98 samples with 4 negatives cannot support a claim about generalization. This is exactly the failure mode that makes published medical ML results collapse in deployment — and identifying it is what motivated Stage 2.

---

# Stage 2 — LSTM

The rebuild fixes all three problems: every patient, no hand-engineered features, and a split that no patient crosses.

### Why an LSTM

The ANN's 7 statistics discard *ordering*. But apnea's signature is inherently temporal — heart rate falls, then rises, in a repeating cycle. A mean and standard deviation cannot distinguish "slow then fast" from "fast then slow." An LSTM consumes the RR interval sequence in order and can learn that cyclic structure directly, no feature design required.

### Data pipeline across 35 patients

Rather than re-detecting R-peaks, Stage 2 uses the database's own expert-verified `.qrs` annotations via `wfdb`:

```python
qrs   = wfdb.rdann(os.path.join(DATA_PATH, patient_id), "qrs")
apnea = wfdb.rdann(os.path.join(DATA_PATH, patient_id), "apn")
```

For each 60-second window, RR intervals are computed and physiologically implausible values are discarded:

```python
rr = rr[(rr >= 0.3) & (rr <= 2.0)]   # 30–200 bpm
```

Anything outside 0.3–2.0 s is a detection artifact, not a heartbeat.

| Stage | Samples |
|---|---|
| All 35 patients processed | 17,006 |
| After dropping segments with < 30 RR intervals | **16,862** |
| Normal | 10,406 |
| Apnea | 6,456 |

The minimum-length filter removes noisy segments where peak detection failed — the raw distribution ran as low as 2 intervals per minute against a median of 63.

### Patient-independent splitting

This is the most important design decision in the project:

```python
unique_patients = np.unique(all_patient_ids)
train_patients, temp_patients = train_test_split(unique_patients, test_size=0.30, random_state=42)
val_patients, test_patients   = train_test_split(temp_patients,  test_size=0.50, random_state=42)
```

The split happens over **patients, not samples**. No minute from a test patient ever appears in training.

| Split | Patients | Samples | Normal | Apnea |
|---|---|---|---|---|
| Train | 24 | 11,470 | 7,067 | 4,403 |
| Validation | 5 | 2,497 | 1,145 | 1,352 |
| Test | 6 | 2,895 | 2,194 | 701 |

Note how different the class balance is in each split — that's the reality of real patients, and it is exactly what a random sample-level split would have hidden.

### Sequence preparation

**Scaling.** `StandardScaler` is fit on **training RR values only**, then applied to validation and test. Fitting on all data would leak test-set statistics into training.

**Padding and masking.** RR sequences vary from 30 to 111 intervals per minute. They're padded to `MAX_LEN = 87` — the 99th percentile of observed lengths, chosen so almost nothing is truncated while the tensor stays compact:

```python
X_train = pad_sequences(X_train, maxlen=87, padding="post",
                        truncating="post", dtype="float32", value=-10.0)
```

The pad value `-10.0` is far outside the range of standardized RR values, and a `Masking(mask_value=-10.0)` layer makes the LSTM skip those timesteps entirely — so padding never influences the hidden state.

**Class balancing.** The training set is undersampled to parity, and class weights are applied on top:

```
Balanced training data — Normal: 4,403 | Apnea: 4,403
class_weights = {0: 0.812, 1: 1.303}
```

### Hyperparameter search

Architecture search with **Keras Tuner** (`RandomSearch`), optimizing **validation AUC** rather than accuracy — AUC is threshold-independent and far more meaningful than accuracy on imbalanced clinical data.

Search space (7 hyperparameters):

| Hyperparameter | Range |
|---|---|
| `num_lstm_layers` | 1 – 3 |
| `lstm_units` (per layer) | 32 – 128, step 32 |
| `lstm_activation` | tanh / relu |
| `lstm_dropout` | 0.1 – 0.5 |
| `dense_units` | 8 – 64, step 8 |
| `dense_activation` | relu / tanh |
| `learning_rate` | 1e-4 – 3e-3 (log scale) |

Weight initialization is set explicitly — `glorot_uniform` for input kernels, `orthogonal` for recurrent kernels (which helps preserve gradient norm through time), and `he_normal` for ReLU dense layers.

Callbacks: `EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)` and `ReduceLROnPlateau(factor=0.5, patience=2)`.

**Best configuration found** (val AUC 0.8145):

```
num_lstm_layers   : 1
lstm_units_0      : 32
lstm_activation_0 : tanh
lstm_dropout_0    : 0.2
dense_units       : 40
dense_activation  : tanh
learning_rate     : 0.00089
```

The search converged on the *smallest* option in almost every dimension — a single 32-unit LSTM layer, **5,713 total parameters**. With ~11k training sequences of a single feature, larger networks overfit.

```
Layer (type)          Output Shape      Param #
─────────────────────────────────────────────────
masking (Masking)     (None, 87, 1)           0
lstm (LSTM)           (None, 32)          4,352
dropout (Dropout)     (None, 32)              0
dense (Dense)         (None, 40)          1,320
dense_1 (Dense)       (None, 1)              41
─────────────────────────────────────────────────
Total params: 5,713 (22.32 KB)
```

### Results on held-out patients

Best model from the tuner, evaluated on the 6 unseen test patients:

```
              precision    recall  f1-score   support

      Normal       0.88      0.45      0.59      2194
       Apnea       0.32      0.81      0.46       701

    accuracy                           0.53      2895
```

**ROC-AUC: 0.7630**

After extended retraining on the full (unbalanced) training set:

```
              precision    recall  f1-score   support

      Normal       0.82      0.94      0.88      2194
       Apnea       0.65      0.36      0.46       701

    accuracy                           0.80      2895
```

Confusion matrix:

```
              Predicted
              Normal  Apnea
Actual Normal   2061    133
       Apnea     449    252
```

### Reading these numbers honestly

The LSTM's 53–80% accuracy looks worse than the ANN's 98%. **It isn't.** They are answering different questions:

| | Stage 1 — ANN | Stage 2 — LSTM |
|---|---|---|
| Patients | 1 | 35 |
| Test samples | 98 | 2,895 |
| Test negatives | 4 | 2,194 |
| Split | Random over minutes | **Patient-independent** |
| Question answered | "Can it classify minutes from a patient it has already seen?" | "Can it classify a patient it has never seen?" |

The second question is the one that matters clinically, and it is dramatically harder. The **ROC-AUC of 0.7630** is the honest headline: well above the 0.50 of chance, and a real signal learned from RR intervals alone across unseen patients.

The two operating points also show a clear precision/recall trade-off. The balanced-training model catches **81% of apnea minutes** at low precision — a screening posture, where a false alarm costs a follow-up and a miss costs a diagnosis. The retrained model is conservative in the other direction. Which is correct depends entirely on whether the tool is a screener or a confirmer.

---

## Known Limitations

- **Single feature stream.** Only RR intervals are used. R-peak amplitude (EDR — ECG-derived respiration) is a well-established complementary signal and is the most obvious next addition.
- **Small patient pool.** 35 recordings, 6 in test. Per-patient performance varies substantially and the test estimate carries wide error bars.
- **No per-patient reporting.** Results are pooled across test patients; a per-recording breakdown would show whether the model fails uniformly or fails on specific individuals.
- **Limited search budget.** The tuner ran 4 trials over a 7-dimensional space — enough to find a reasonable configuration, not enough to claim it is the best one.
- **Segment-level only.** Clinical diagnosis uses AHI (apnea-hypopnea index) aggregated across a whole night; this model classifies isolated minutes and does not yet roll up to a per-patient diagnosis.

---

## Tech Stack

| Category | Tools |
|---|---|
| Signal processing | wfdb, SciPy (`find_peaks`), NumPy |
| Feature engineering | SciPy stats (kurtosis, skew) |
| Deep learning | TensorFlow / Keras |
| Tuning | Keras Tuner (`RandomSearch`) |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |
| Data access | kagglehub |

---

## Running It

```bash
pip install wfdb kagglehub keras-tuner tensorflow scikit-learn scipy matplotlib pandas
```

Run the notebooks in order — Stage 1 establishes the baseline and the problem, Stage 2 is the rebuild:

1. `sleepapenaannproject.ipynb` — ANN on engineered features
2. `LSTM_sleep_apnea 1.ipynb` — LSTM on raw RR sequences

The Apnea-ECG database downloads automatically via `kagglehub`.

---

## What This Project Demonstrates

- **Biomedical signal processing** end to end — raw ECG → R-peak detection → RR intervals → physiological artifact filtering
- **Feature engineering vs. representation learning**, implemented as a direct comparison rather than an assertion
- **Patient-independent validation** — recognizing that random splits leak in patient data, and rebuilding the protocol to prevent it
- **Variable-length sequence handling** with percentile-based padding and explicit masking
- **Class imbalance handling** through undersampling and class weighting together
- **Metric selection under imbalance** — tuning on AUC rather than accuracy, and reading precision/recall as an operating-point choice
- **Diagnosing an inflated result**, which is the harder and more valuable skill than producing one

---

## Disclaimer

Research and educational project. Not a medical device; not for clinical diagnosis.
