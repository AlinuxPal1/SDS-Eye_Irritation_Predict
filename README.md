# SDS-EYE Predictor  
Predictive Safety Screening for Eye Irritation in SDS Ingredients

---

## Overview
**SDS-EYE Predictor** is a lightweight, fully reproducible project designed to classify potential **eye irritation risk** of chemical ingredients commonly found in **Safety Data Sheets (SDS)**.

The project uses:
- **RDKit** for 1024-bit Morgan fingerprints  
- A pre-trained **XGBoost classifier** (not included in the repository)  
- A clean and auditable Python pipeline  
- Optional applicability-domain (DA) checks based on Tanimoto similarity  

The goal is to demonstrate:
- Transparent data processing  
- Reproducible model loading and prediction  
- Clean, industry-aligned design  
- Practical SDS-style usage  

The project ideal for **portfolio**, **LinkedIn**, and **early ML-safety R&D demonstrations**.

---

## Project Structure

sds-eye-predictor/
│
├── src/
│   └── sds_eye/
│       ├── featurization.py
│       ├── model_eye.py
│       └── sds_eye_predict.py
│
├── data/
│   └── test_sds_eye.csv
│
├── models/
│   └── eye_xgb_1024_scaffold.pkl   (not included)
│
├── reports/
│
├── run_sds_eye_demo.py
├── requirements.txt
└── README.md

---

## Installation  

Create an environment and install dependencies:

```bash
pip install -r requirements.txt

