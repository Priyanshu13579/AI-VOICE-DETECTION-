p# AI Voice Detection

Detects whether a `.wav` audio is Real (human) or Fake (AI/cloned) using classical audio features and an SVM classifier. Includes a CLI prompt (`predict.py`) and a simple Flask web UI (`main.py`).

## Overview
- Input: `.wav` files uploaded or provided via path.
- Features: MFCCs + deltas, spectral centroid/rolloff/contrast, chroma, zero-crossing rate, RMS energy; aggregated by mean and std.
- Model: SVM (RBF kernel) with `class_weight=balanced`, probability enabled, and standardized features via `StandardScaler`.
- Artifacts: `model.pkl` (classifier) and `scaler.pkl` (feature scaler).
- Web: Upload an audio file, get classification with confidence.

## Project Structure
- `train.py`: Training pipeline (feature extraction, scaling, SVM fit, evaluation, model save).
- `predict.py`: Loads artifacts, extracts features, predicts label + confidence.
- `main.py`: Flask app for upload + result page.
- `dataset/real`: Human voice training clips.
- `dataset/fake`: AI/cloned voice training clips.
- `dataset/test`: Optional test clips.
- `uploads/`: Temporary storage for files uploaded via web.

## Pipeline
1. Data Collection
   - Place human recordings in `dataset/real` and AI/cloned voices in `dataset/fake`.
2. Feature Extraction (in `train.extract_mfcc_features()`)
   - Resample to 16 kHz mono for consistency.
   - Compute 20 MFCCs and their deltas (1st and 2nd order).
   - Compute spectral features: centroid, rolloff, contrast.
   - Compute prosodic/energy features: zero-crossing rate (ZCR), RMS.
   - Compute chroma (pitch class energy).
   - Aggregate each feature set over time using mean and std.
3. Preprocessing
   - Fit `StandardScaler` on training features; transform train/test.
4. Model
   - `SVC(kernel='rbf', gamma='scale', C=1.0, class_weight='balanced', probability=True)`.
5. Evaluation
   - `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`.
   - Report `accuracy_score` and `confusion_matrix`.
6. Persistence
   - Save model to `model.pkl` and scaler to `scaler.pkl` with `joblib.dump()`.

## Model & Features
- Why MFCCs: capture shape of human vocal tract; widely used in speech tasks.
- Deltas: capture dynamics (how MFCCs change over time).
- Spectral features: distinguish natural vs synthetic spectral patterns.
- ZCR/RMS: detect periodicity/noise and energy variations.
- Chroma: summarizes pitch content; cloned voices may show atypical stability.
- SVM (RBF): effective for non-linear separation with small/medium datasets; `class_weight='balanced'` helps with imbalances.

## Results
- With 34 Real and 34 Fake samples included, a stratified 80/20 split yielded:
  - Accuracy: 1.0
  - Confusion Matrix: `[[7, 0], [0, 7]]`
- Note: Perfect accuracy on small splits can be optimistic. Avoid data leakage (no near-duplicate clips across train/test), and prefer more data or k-fold validation for robust estimates.

## Prediction Flow
Implemented in [`predict.py`](predict.py):
- Load `model.pkl` and `scaler.pkl`.
- Validate path and extension.
- Extract features using the same function as training.
- Scale features and predict label (`0 → Real`, `1 → Fake`).
- If available, compute `predict_proba` for confidence % and include in the output string.

## Web App Flow
Implemented in [`main.py`](main.py):
- `GET /`: Render `index.html` to upload a `.wav` file.
- `POST /result`: Validate file, save to `uploads/`, call `analyze_audio()`, render result.
- Handles errors (missing file, invalid type) using `flash()` messages.

## Usage
### Install Dependencies (Windows PowerShell)
```powershell
pip install librosa numpy scikit-learn joblib flask werkzeug
```

### Train
```powershell
python train.py
```
- Saves `model.pkl` and `scaler.pkl` in the project root.

### Predict (CLI)
```powershell
python predict.py
```
- Enter the full path to a `.wav` file when prompted.

### Run Web App
```powershell
python main.py
```
- Open http://127.0.0.1:5000 and upload a `.wav` file.

## File-by-File Summary
- [`train.py`](train.py)
  - `extract_mfcc_features(audio_path, ...)`: Build comprehensive feature vector (MFCCs+deltas + spectral + chroma + ZCR + RMS, mean/std aggregation).
  - `create_dataset(directory, label)`: Load `.wav`s from a folder, extract features, assign labels.
  - `train_model(X, y)`: Split, scale, fit SVM, evaluate, save artifacts.
  - `main()`: Prepare datasets from `dataset/real` and `dataset/fake`, stack and call `train_model`.
- [`predict.py`](predict.py)
  - `analyze_audio(input_audio_path)`: Load artifacts, extract + scale features, predict label and confidence, return text result.
- [`main.py`](main.py)
  - Flask routes for upload and result display; stores uploads; calls `analyze_audio()`.

## Tips
- Data diversity matters: add real samples from different mics, languages, and environments, plus varied AI voices.
- Keep train/test separation clean to avoid leakage.
- If many false positives/negatives occur, retrain after adding representative samples.
- For stronger robustness, consider k-fold CV, feature selection, or ensemble methods.

