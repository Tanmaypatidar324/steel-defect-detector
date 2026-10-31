# AI Steel Surface Defect Detector
AI Manthan 2.0 | PS 4: Real-Time AI Quality Assurance and Defect Detection

## Problem
Manual inspection of steel surfaces is slow and error-prone.

## Solution
A YOLOv8 classification model detects 6 steel surface defects
(crazing, inclusion, patches, pitted surface, rolled-in scale, scratches)
with 99.3% validation accuracy on the NEU dataset. A Streamlit dashboard
shows defect type, severity, recommended action, an AI-generated report
(Gemini) and a live defect counter with chart.

## Tech Stack
YOLOv8 (Ultralytics), PyTorch, Streamlit, Gemini API, Python

## How to Run
pip install -r requirements.txt
python -m streamlit run app.py


## Limitations
Closed-set classifier trained on clean NEU data; real factory images may
perform differently. Future work: bounding-box detection, live webcam,
"not steel" class, edge deployment.

## Team
Team Zenith