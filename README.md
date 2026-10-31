# SteelScan AI - Steel Surface Defect Detector
AI Manthan 2.0 | PS 4: Real-Time AI Quality Assurance and Defect Detection

## Problem
Manual inspection of steel surfaces is slow, inconsistent and error-prone.

## Solution
A YOLOv8 classification model detects 6 steel surface defects (crazing, inclusion,
patches, pitted surface, rolled-in scale, scratches) with 99.3% validation accuracy
on the NEU dataset. A Streamlit dashboard turns the prediction into a factory decision.

## Features
- **3 input modes:** image upload, camera snapshot, live video / webcam stream
- **Factory verdict:** REJECT / REWORK / PASS WITH NOTE based on severity, or MANUAL CHECK when the model is unsure
- **Explainable AI:** Grad-CAM heatmap and approximate defect region show where the model looked
- **Top-3 predictions** with confidence bars
- **AI inspection report** (Gemini) in **English or Hindi**, with an offline fallback
- **Alerts:** red banner + sound on High severity, and a production-line alert when 3 of the last 10 items are High severity
- **Live dashboard:** item counter, defects by type, cumulative trend by severity, inspection log
- **Exports:** one-page PDF report per item and CSV inspection log
- **ROI calculator:** estimated labour and quality savings (editable assumptions)
- **Non-steel rejection** filter for images that are clearly not steel
- **Model evaluation script** (`evaluate.py`): confusion matrix, per-class precision/recall, robustness to blur / noise / lighting, PyTorch vs ONNX latency

## Tech Stack
YOLOv8 (Ultralytics), PyTorch, Streamlit, OpenCV, Gemini API, Python

## How to Run
```
pip install -r requirements.txt
python -m streamlit run app.py
```
Keep `app.py`, `helpers.py` and `best.pt` in the same folder.

Optional: set the Gemini key as environment variable `GEMINI_API_KEY`
(or in `.streamlit/secrets.toml`) instead of typing it in the sidebar.

## Evaluate the model
```
python evaluate.py --data path/to/val
```
`path/to/val` must contain one folder per class. Results are saved in `metrics/` and
appear automatically in the app under "Model performance and robustness".

## Limitations
- Closed-set classifier trained on clean NEU data; real factory images may perform differently
  (see the robustness chart produced by `evaluate.py`).
- The Grad-CAM region box is an approximation, not a trained bounding-box detector.
- The non-steel filter is a simple colour heuristic.
- ROI numbers are assumptions that must be replaced with real plant data.
- Webcam mode needs a camera on the machine running the app (not available on cloud hosting; use a video file there).

## Future Work
Bounding-box detection (YOLOv8 detect on NEU-DET), edge deployment (ONNX / TensorRT),
a trained "not steel" class, integration with PLC / conveyor reject mechanisms.

## Team
Team Zenith