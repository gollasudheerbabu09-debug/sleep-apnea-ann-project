# Sleep Apnea Detection from Single-Lead ECG: ANN → LSTM

**Live site with an in-browser demo: https://gollasudheerbabu09-debug.github.io/sleep-apnea-ann-project/**

Detecting obstructive sleep apnea, minute by minute, from the timing of heartbeats alone. Built in two stages: a **feature-based ANN** on a single patient, then a **sequence-based LSTM** across 35 patients, evaluated on people the model has never seen.

The second stage exists because of what the first revealed. That progression, and what it exposed about evaluating medical ML honestly, is the core of this project.

## Results at a glance

| Model | Trained on | Tested on | ROC-AUC |
|---|---|---|---|
| Stage 1: ANN, 7 RR statistics | 1 patient (a01) | other minutes of the same night | 1.000 (accuracy 98.98%) |
| Stage 1b: same ANN | 24 patients | 6 unseen patients | **0.715** |
| Stage 2: LSTM on raw RR sequences | 24 patients | 6 unseen patients | **0.780** |

- **Stage 1's 98.98% is inflated.** One patient, a random split, and only 4 normal minutes in the test set; always answering "apnea" already scores about 96%.
- **The LSTM beats the feature-based ANN on exactly the same patients** (0.780 vs 0.715). Reading heartbeats in order is worth about +0.065 AUC over summary statistics.
- The decision threshold (0.356) was chosen on the 5 validation patients, never on the test set.

ROC curves, confusion matrices, training curves and a patient-by-patient breakdown are on the [live site](https://gollasudheerbabu09-debug.github.io/sleep-apnea-ann-project/), where you can also run the trained models on real test-patient minutes or your own RR intervals.

---

## Background

Sleep apnea is diagnosed with polysomnography, an overnight lab study that is expensive and capacity-limited. ECG offers a cheaper screening route: during an apnea the heart slows, and when breathing resumes it speeds up. Repeated every 30 to 90 seconds, this produces **cyclic variation in heart rate** in the **RR interval series**, the time between consecutive R-peaks. Both models learn from that series.

Every minute of the **PhysioNet Apnea-ECG Database** is labelled `A` (apnea) or `N` (normal) by experts, so the task is per-minute binary classification.

| Property | Value |
|---|---|
| Recordings | 35 (`a01`–`a20` apnea, `b01`–`b05` borderline, `c01`–`c10` control) |
| Sampling rate | 100 Hz |
| Labels | one per 60-second segment |
| Usable minutes | 16,862 (10,406 normal, 6,456 apnea) |

---

## Stage 1: ANN on engineered features (one patient)

Raw ECG from record `a01` is cut into 60-second windows. R-peaks are found with `scipy.signal.find_peaks(distance=60, prominence=0.5)`: the 0.6 s minimum spacing stops double-counting a beat, and the prominence threshold selects the R spike over P and T waves. Each window's RR series becomes **7 statistics** (max, min, mean, median, standard deviation, kurtosis, skewness), classified by a dense network (7 → 40 → 20 → 1, 1,161 parameters).

**Why the near-perfect score is not the story.** Record `a01` has 470 apnea minutes against 19 normal. Three problems stack up: **one patient** (the network may learn one heart, not apnea), **severe imbalance** (accuracy is nearly meaningless at 96/4), and a **random split over minutes** (neighbouring minutes of the same night land on both sides). A test set with 4 negative examples cannot support a claim about anyone else. Recognising that motivated Stage 2.

## Stage 1b: the fair comparison

The same 7 features and the same network, trained and tested on exactly the LSTM's 35-patient split. This isolates the one question the original project could not answer: is reading beats *in order* actually better than summarising them? On unseen patients the ANN reaches **AUC 0.715**.

## Stage 2: LSTM on raw RR sequences (35 patients)

**Why an LSTM.** Seven statistics discard ordering, but apnea's signature is temporal: heart rate falls, then rises. A mean and standard deviation cannot tell "slow then fast" from "fast then slow".

**Data.** The database's expert-verified `.qrs` R-peak annotations are used. RR intervals outside 0.3–2.0 s (30–200 bpm) are dropped as artifacts, and minutes with fewer than 30 intervals are dropped as detection failures.

**Patient-independent split**, the most important decision in the project. Records are split before any minute is touched, so no test patient's minute is ever seen in training.

| Split | Records | Minutes | Normal | Apnea |
|---|---|---|---|---|
| Train | 24 | 11,470 | 7,067 | 4,403 |
| Validation | 5 (`a09 a13 a16 a20 b02`) | 2,497 | 1,145 | 1,352 |
| Test | 6 (`a10 a14 a17 b05 c02 c05`) | 2,895 | 2,194 | 701 |

**Sequences.** A scaler fit on training RR values only; each minute post-padded to 87 intervals (the 99th percentile) with value −10, and a `Masking` layer so padding never affects the hidden state.

**Model search.** Keras Tuner random search over LSTM layers (1–3), units (32–128), activations, dropout, dense units and learning rate, scored by **validation AUC**. The best configuration is retrained with early stopping on validation AUC and learning-rate reduction; validation AUC levelled off around 0.82.

**Result on six unseen patients: ROC-AUC 0.780.** Well above chance from heartbeat timing alone, on people the model has never met.

---

## Audit: bugs found and fixed in the original notebooks

1. **Labels shifted by one minute (Stage 1).** The annotation file was read with its first line treated as a header, so every ECG window was paired with the *next* minute's label (488 labels instead of 489). Labels now come from the `.apn` annotations, each window tied to its own minute.
2. **The "retrained" LSTM was never evaluated.** The notebook retrained `best_model` but predicted with a different variable, `model`. Its reported 80% accuracy came from that other model, with an AUC of only 0.58. Now one object is trained, evaluated, saved and exported, and the exported weights are verified against Keras.
3. **Imbalance corrected twice.** Training data was undersampled to 50/50 *and* weighted with class weights from the original 61/39 split, pushing the model toward "apnea". Now all training minutes are kept with balanced class weights applied once. Together with the fixes above, test AUC rose from 0.763 to 0.780.

Smaller fixes: the Stage 1 scaler was fit before the split (test leakage); skipped windows could desynchronise features and labels; the threshold is chosen on validation patients instead of fixed at 0.5.

---

## Repository

```
sleep-apnea-ann-project/
├── .github/workflows/pages.yml     # deploys the website on every push
├── sleep-apnea-ecg/                # the project
│   ├── src/apnea/                  #   pipeline package: data, features, splits, models, evaluation, export
│   ├── scripts/run_pipeline.py     #   runs every stage and writes the site's data
│   ├── tests/test_pipeline.py      #   fast tests, no TensorFlow or download needed
│   ├── docs/                       #   the website (HTML/CSS/JS); docs/data holds results and model weights
│   ├── notebooks/                  #   original notebooks + a Kaggle/Colab runner
│   └── requirements.txt
├── sleep-apena-ann-project.ipynb   # original Stage 1 notebook
└── LSTM_sleep_apnea (1).ipynb      # original Stage 2 notebook
```

The website is static: the trained LSTM and ANN are exported to JSON and run in the visitor's browser, so there is no server. A test checks that the browser implementation matches the Python model.

## Running it

```bash
git clone https://github.com/gollasudheerbabu09-debug/sleep-apnea-ann-project.git
cd sleep-apnea-ann-project/sleep-apnea-ecg
pip install -r requirements.txt

python -m pytest                          # fast checks
python scripts/run_pipeline.py --quick    # smoke test, a few minutes
python scripts/run_pipeline.py            # full run
```

The dataset is located automatically (an attached Kaggle dataset, `kagglehub`, or a direct PhysioNet download). Without a local GPU, use `sleep-apnea-ecg/notebooks/03_run_pipeline_kaggle_colab.ipynb` on Kaggle with a GPU; a full run takes about 15 minutes. The run writes `docs/data/results.json` and `docs/data/model.json`; committing them updates the site.

## Limitations

- **One signal.** Only RR intervals. ECG-derived respiration from R-peak amplitude is the obvious next input.
- **Few patients.** 35 recordings, 6 in test, so every number carries wide error bars.
- **Minutes, not diagnoses.** Clinical diagnosis uses the apnea-hypopnea index from a full sleep study. The per-patient roll-up on the site approximates the idea; it is not a substitute.
- **Small search.** A handful of random-search trials finds a reasonable configuration, not the best one.

## Tech stack

wfdb, SciPy, NumPy, scikit-learn, TensorFlow/Keras, Keras Tuner, plain HTML/CSS/JavaScript, GitHub Pages, GitHub Actions.

## Data

Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. *The Apnea-ECG Database.* Computers in Cardiology 2000;27:255–258. Available on PhysioNet.

## Disclaimer

Research and educational project. Not a medical device; not for clinical diagnosis.
