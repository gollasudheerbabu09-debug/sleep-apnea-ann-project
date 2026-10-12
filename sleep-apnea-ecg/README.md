# Sleep Apnea Detection from Single-Lead ECG: ANN → LSTM

[![tests](https://github.com/YOUR-USERNAME/sleep-apnea-ecg/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR-USERNAME/sleep-apnea-ecg/actions/workflows/tests.yml)

**Live site and in-browser demo: [https://YOUR-USERNAME.github.io/sleep-apnea-ecg/**](https://gollasudheerbabu09-debug.github.io/sleep-apnea-ann-project/#run)

Detecting obstructive sleep apnea, minute by minute, from the timing of heartbeats alone. Built in two stages: a **feature-based ANN** on a single patient, then a **sequence-based LSTM** across 35 patients evaluated on people the model has never seen.

The second stage exists because of what the first revealed. That progression, and what it exposed about evaluating medical ML honestly, is the core of this project. A line-by-line audit of the original notebooks then turned up three bugs that changed the reported numbers; they are fixed here and documented below.

---

## Background

Sleep apnea is diagnosed with polysomnography, an overnight lab study that is expensive and capacity-limited. ECG offers a cheaper screening route: during an apnea the heart slows, and when breathing resumes on arousal it speeds up. Repeated every 30 to 90 seconds, this produces **cyclic variation in heart rate** that lives in the **RR interval series**, the time between consecutive R-peaks. Both models learn from that series.

Every minute of the **PhysioNet Apnea-ECG Database** is labelled `A` (apnea) or `N` (normal) by experts, so the task is per-minute binary classification.

| Property | Value |
|---|---|
| Recordings | 35 (`a01`–`a20` apnea, `b01`–`b05` borderline, `c01`–`c10` control) |
| Sampling rate | 100 Hz |
| Labels | one per 60-second segment |
| Usable minutes | 16,862 (10,406 normal, 6,456 apnea) |

---

## Repository

```
.
├── src/apnea/                 # the pipeline as a small Python package
│   ├── config.py              #   every constant in one place
│   ├── data.py                #   download, R-peaks, RR intervals, artifact filtering
│   ├── features.py            #   the 7 RR statistics (Stage 1)
│   ├── splits.py              #   patient-independent split
│   ├── sequences.py           #   scaling, padding, masking (Stage 2)
│   ├── models.py              #   ANN, LSTM and Keras Tuner search space
│   ├── evaluate.py            #   AUC, thresholds, per-patient roll-up
│   └── export.py              #   Keras → JSON, plus a NumPy forward pass used to verify it
├── scripts/run_pipeline.py    # runs every stage, writes everything the site shows
├── tests/test_pipeline.py     # fast tests: no TensorFlow or download needed
├── docs/                      # the website (GitHub Pages)
│   ├── index.html, css/, js/  #   static site; js/model.js runs the trained networks in the browser
│   └── data/                  #   results.json and model.json, written by the pipeline
├── notebooks/
│   ├── 01_ann_single_patient_original.ipynb
│   ├── 02_lstm_35_patients_original.ipynb
│   └── 03_run_pipeline_kaggle_colab.ipynb
└── requirements.txt
```

The two original notebooks are kept unchanged as the record of the first version. The `src/` package is the corrected rewrite.

---

## Stage 1: ANN on engineered features (one patient)

Raw ECG from record `a01` is cut into 60-second windows. R-peaks are found with `scipy.signal.find_peaks(distance=60, prominence=0.5)`: a 0.6 s minimum spacing stops the detector firing twice on one beat, and the prominence threshold picks the tall R spike over the P and T waves. Each window's RR series is collapsed into **7 statistics**: max, min, mean, median, standard deviation, kurtosis and skewness. A dense network (7 → 40 → 20 → 1, **1,161 parameters**) classifies them.

### Result, and why it isn't the story

| Metric | Original notebook |
|---|---|
| Test accuracy | 97.96% |
| Always predicting "apnea" | 95.9% |
| Normal minutes caught | 2 of 4 |

Record `a01` is a severe apnea patient: 470 apnea minutes against 19 normal. Three problems stack up: **one patient** (the network may have learned one heart, not apnea), **severe imbalance** (accuracy is nearly meaningless at 96/4), and a **random split over minutes** (neighbouring minutes from the same night sit on both sides). A test set with 4 negative examples cannot support a claim about anyone else. Recognising that motivated Stage 2.

---

## Stage 2: LSTM on raw RR sequences (35 patients)

**Why an LSTM.** Seven statistics discard ordering, but apnea's signature is temporal: heart rate falls, then rises. A mean and standard deviation cannot tell "slow then fast" from "fast then slow". An LSTM reads the beats in order.

**Data.** Instead of re-detecting peaks, Stage 2 uses the database's expert-verified `.qrs` annotations. RR intervals outside 0.3–2.0 s (30–200 bpm) are discarded as artifacts, and minutes with fewer than 30 intervals are dropped as detection failures.

**Patient-independent split** (the most important decision in the project). Records are split 24 / 5 / 6 before any minute is touched; a test patient's minutes never appear in training. A test asserts the split exactly matches the original notebook's.

| Split | Records | Minutes | Normal | Apnea |
|---|---|---|---|---|
| Train | 24 | 11,470 | 7,067 | 4,403 |
| Validation | 5 | 2,497 | 1,145 | 1,352 |
| Test | 6 (`a10 a14 a17 b05 c02 c05`) | 2,895 | 2,194 | 701 |

**Sequences.** A scaler fit on training RR values only; sequences post-padded to 87 (the 99th percentile of intervals per minute) with value −10 and a `Masking` layer so padding never touches the hidden state.

**Search.** Keras Tuner random search over layer count (1–3), units (32–128), activations, dropout, dense units and learning rate, scored by **validation AUC**. The original search converged on the smallest options: one 32-unit LSTM, **5,713 parameters**.

### Result on six unseen patients

Original notebook, tuner's best model, threshold 0.5:

| | Precision | Recall |
|---|---|---|
| Normal | 0.88 | 0.45 |
| Apnea | 0.32 | 0.81 |

**ROC-AUC 0.763.** Well above chance from heartbeat timing alone, on people the model has never seen. That is the honest headline, and it answers a much harder question than Stage 1's 98%.

> After running `scripts/run_pipeline.py`, the corrected numbers, ROC curves, per-patient results and the threshold-tuned operating point appear on the website. Paste them here too.

---

## Audit: bugs found in the original notebooks

1. **Labels shifted by one minute (Stage 1).** `a01.txt` was read with `pd.read_csv` and its first line became the header, so minute 0's label vanished and every window was paired with the *next* minute's label (488 labels instead of 489). Fixed by reading labels from the `.apn` annotations and keeping each window tied to its own index.
2. **The "retrained" LSTM was never evaluated.** The notebook retrained `best_model`, then predicted with a different variable, `model`. The reported "80% accuracy" came from that other model, whose AUC was **0.58**. Now one object is trained, evaluated, saved and exported, and the exported weights are checked against Keras before being written.
3. **Imbalance corrected twice.** Training data was undersampled to 50/50 *and* weighted with class weights computed from the original 61/39 split, pushing the model toward "apnea" (81% recall at 32% precision). Fixed: all training minutes, balanced class weights once.

Smaller fixes: the Stage 1 scaler was fit before the split (test leakage); skipped windows could desynchronise features and labels; the threshold is now chosen on validation patients (Youden's J) instead of fixed at 0.5. A new **Stage 1b** trains the Stage 1 ANN on exactly the LSTM's 35-patient split, which makes "features vs. sequences" a fair comparison for the first time.

---

## Running it

```bash
git clone https://github.com/YOUR-USERNAME/sleep-apnea-ecg.git
cd sleep-apnea-ecg
pip install -r requirements.txt

python -m pytest                      # fast checks, seconds
python scripts/run_pipeline.py --quick   # smoke test, a few minutes
python scripts/run_pipeline.py           # full run
```

The data is found automatically: `$APNEA_DATA`, an attached Kaggle dataset (`ecerulm/apneaecg`), `kagglehub`, or a direct PhysioNet download (~200 MB) into `data/`. A full run takes roughly 10–20 minutes on a GPU and longer on a laptop CPU; `--trials` and `--epochs` trade time for search depth. No local GPU? Use `notebooks/03_run_pipeline_kaggle_colab.ipynb`.

The run writes `docs/data/results.json`, `docs/data/model.json` and `models/*.keras`. Commit the `docs/` changes and the website updates.

## Deploying the website

The site is static and needs no server: the trained networks run in the visitor's browser from `docs/data/model.json`.

1. Push this repository to GitHub.
2. **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/docs` → Save.**
3. After about a minute it is live at `https://YOUR-USERNAME.github.io/sleep-apnea-ecg/`. The GitHub link and clone command on the page fill themselves in from that address.

Until the pipeline has been run once, the site shows the numbers recorded in the original notebooks (clearly marked) and the demo panel explains how to switch it on.

---

## Limitations

- **One signal.** Only RR intervals. ECG-derived respiration (from R-peak amplitude) is the obvious next input.
- **Few patients.** 35 recordings, 6 in test; every number carries wide error bars.
- **Minutes, not diagnoses.** Clinical diagnosis uses the apnea-hypopnea index from a full sleep study. The per-patient "apnea minutes per hour" roll-up approximates the idea; it is not a substitute.
- **Small search.** A handful of random-search trials over seven dimensions finds a reasonable configuration, not the best one.

## Tech stack

`wfdb` · SciPy · NumPy · scikit-learn · TensorFlow/Keras · Keras Tuner · plain HTML/CSS/JS (no build step, no framework) · GitHub Pages · GitHub Actions

## Data and citation

Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. *The Apnea-ECG Database.* Computers in Cardiology 2000;27:255–258. Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 2000;101(23):e215–e220.

## Disclaimer

Research and educational project. Not a medical device; not for clinical diagnosis.
