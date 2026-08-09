# SignBridge-ML

Real-time American Sign Language (ASL) fingerspelling recognition from a
webcam, using MediaPipe hand landmarks and a scikit-learn classifier.

> **Note on scope**: this project originally targeted a small vocabulary
> of Indian Sign Language (ISL) words. It was changed to the **ASL
> fingerspelling alphabet** because ISL words are typically dynamic,
> multi-frame, sometimes two-handed signs, while this pipeline is built
> around classifying a **single static frame** of one hand's landmarks.
> The ASL alphabet (minus J and Z, see below) is a much better fit for
> that architecture: nearly every letter is a distinct, held, one-hand
> pose.

## Vocabulary

`configs/config.yaml` -> `vocabulary` lists the 24 letters this project
classifies: `a-i, k-y` (excludes `j` and `z`).

**J and Z are deliberately excluded.** In real ASL fingerspelling, J and
Z are *traced* with hand motion (J curves, Z zig-zags) rather than held
as a single pose. A classifier trained on one static landmark snapshot
per sample has no way to represent that motion, so including them would
just teach the model an arbitrary mid-motion frame that doesn't
generalize. If you want them anyway (e.g. approximated from one frame),
add `j` and `z` back into `vocabulary` in `config.yaml` -- no code
changes needed, since the vocabulary list drives everything downstream.

## Project layout

```
SignBridge-ML/
├── app.py                              # main entry point (real-time prediction)
├── configs/config.yaml                 # all paths, vocabulary, model & camera settings
├── dataset/
│   ├── raw/<letter>/sample_NNNN.npy    # raw (21,3) hand landmarks, one file per sample
│   └── processed/{train,val,test}.csv  # normalized (63,) feature vectors + label
├── models/
│   ├── checkpoints/                    # every trained model, timestamped
│   └── final/model.joblib              # best model so far, what app.py loads
├── outputs/
│   ├── confusion_matrix/               # evaluate.py's confusion matrix PNGs
│   ├── logs/                           # predict.py / app.py run logs
│   └── predictions/                    # reserved for future use
└── src/
    ├── data/
    │   ├── collect_data.py             # interactive webcam data collection
    │   ├── convert_images_to_landmarks.py  # batch: images -> raw .npy landmarks
    │   └── preprocess.py               # raw .npy -> normalized train/val/test CSVs
    ├── inference/
    │   ├── mediapipe_detector.py       # MediaPipe Hands wrapper
    │   └── predict.py                  # real-time webcam prediction loop
    ├── models/
    │   ├── train_model.py              # trains + promotes the best model
    │   └── evaluate.py                 # accuracy/precision/recall/F1 + confusion matrix
    └── utils/
        └── utils.py                    # config loading, normalization, model loading, logging
```

## Setup

```bash
pip install -r requirements.txt
```

If `HandDetector()` fails with `AttributeError: module 'mediapipe' has
no attribute 'solutions'`, your mediapipe build shipped without the
legacy Solutions API this project depends on -- see the note at the
bottom of `requirements.txt` for a known-working pinned version.

## Getting training data

You need raw `(21, 3)` hand-landmark samples under `dataset/raw/<letter>/`
before anything else will work. Two ways to get them -- **mix and match
freely**, since `preprocess.py` only cares that the files exist, not
which tool produced them:

### Option A: convert an existing image dataset (recommended, no webcam needed)

Download an ASL alphabet image dataset where images are organized one
folder per letter (e.g. Kaggle's *"American Sign Language A-Z Dataset +
Hand Landmarks"*). Rename folders to lowercase single letters if needed
so they match `vocabulary` in `config.yaml` (e.g. `A/` -> `a/`).

```bash
python src/data/convert_images_to_landmarks.py --input_dir /path/to/downloaded/dataset
```

This runs MediaPipe over every image and writes `.npy` landmark files in
the same format `collect_data.py` produces. Images where no hand is
detected are skipped with a warning, not an error. Try it on a small
subset first (a couple of letters, ~10 images each) to sanity-check
detection rates before converting a full dataset.

### Option B: collect your own samples with a webcam

```bash
python src/data/collect_data.py --sign a
python src/data/collect_data.py --sign b
# ... repeat for each letter in vocabulary
```

Press SPACE to capture a sample when "HAND DETECTED" shows, `q` to stop.
Sample count per run defaults to `data_collection.samples_per_sign` in
`config.yaml`.

## Running the pipeline

Once `dataset/raw/<letter>/` has samples for your vocabulary:

```bash
python src/data/preprocess.py        # -> dataset/processed/{train,val,test}.csv
python src/models/train_model.py     # -> models/final/model.joblib
python src/models/evaluate.py        # accuracy/F1/confusion matrix on the test set
python app.py                        # real-time webcam prediction
```

`train_model.py` only promotes a checkpoint to `models/final/model.joblib`
if it beats the previous best validation accuracy (tracked in
`models/final/best_meta.json`), so re-running training is always safe.

## Configuration

Everything path- and setting-related lives in `configs/config.yaml`:
vocabulary, dataset split ratios, camera device/resolution, MediaPipe
detection thresholds, model type/hyperparameters (`random_forest` /
`svm` / `mlp`), and inference settings (confidence threshold, frame
smoothing window).

**YAML gotcha to know about**: unquoted `yes`/`no`/`true`/`false`/`on`/`off`
in YAML 1.1 are parsed as booleans, not strings. If you ever add a sign
whose name collides with one of those words, quote it (e.g. `"yes"`).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `app.py` says no model found | Run `train_model.py` first (which needs `preprocess.py` output, which needs raw samples) |
| `preprocess.py` says "No data found" | `dataset/raw/<letter>/` folders are empty -- collect or convert samples first |
| Low detection rate in `convert_images_to_landmarks.py` | Try lowering `mediapipe.min_detection_confidence` in `config.yaml`, or check the images actually show a clear single hand |
| Webcam won't open | Check `camera.device_index` in `config.yaml` matches your actual camera |
