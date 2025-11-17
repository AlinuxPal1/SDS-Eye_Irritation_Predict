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

The goal is **not** to build the best model in the world, but to demonstrate:
- Transparent data processing  
- Reproducible model loading and prediction  
- Clean, industry-aligned design  
- Practical SDS-style usage  

This makes the project ideal for **portfolio**, **LinkedIn**, and **early ML-safety R&D demonstrations**.

---

## Project Structure
