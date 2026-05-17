# PhysioGuide LSTM Training

This backend trains one 4-class sequence model:

- `chin_tuck_correct`
- `chin_tuck_incorrect`
- `shoulder_abduction_correct`
- `shoulder_abduction_incorrect`

Each CSV row must contain 10 frames of 10 MediaPipe pose landmarks, followed by `label` and `exercise`.

## Setup

Use a Python environment where TensorFlow installs successfully. If `python` or the Windows `py` launcher points to a broken or unsupported interpreter, install/use another Python version and activate it first.

```powershell
cd "C:\Users\Dell 7400\PhysioGuide\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Validate Data

Run this before training. It checks CSV columns, labels, feature shapes, grouped splits, and preprocessing without importing TensorFlow.

```powershell
python train_lstm.py --prepare-only
```

## Train

```powershell
python train_lstm.py --epochs 120 --batch-size 32
```

Artifacts written in this folder:

- `exercise_lstm_model.keras`
- `exercise_scaler.json`
- `exercise_label_map.json`
- `training_metrics.json`
- `classification_report.csv`
- `confusion_matrix.csv`
- `training_log.csv`
- `training_curves.png` if matplotlib is installed

## For Better Real-World Accuracy

Record correct and incorrect reps from several people, distances, lighting conditions, and camera angles. Keep wrong-form examples specific: shoulder shrug, bent elbow, too high, too low, fast/uncontrolled lowering, chin flexion, chin extension, and not enough retraction.

For chin tuck, the current CSVs do not include nose or ear landmarks, so the LSTM cannot directly see head/neck motion. The server still uses MediaPipe nose/ear rules for feedback, but the next dataset version should add nose, left/right ear, and maybe mouth landmarks to make the model much stronger for chin tuck correctness.
