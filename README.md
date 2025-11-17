***Model Status (Version 0.1 – Prototype Release)***

This repository currently includes a prototype version of the SDS-EYE predictive model.
The full software pipeline (featurization, batch prediction, CLI, reporting) is functional and stable, but the underlying ML model is not yet validated for real-world toxicological use.

Known limitations (v0.1):
	•	The model was trained on a limited and chemically homogeneous dataset, which causes:
	•	overprediction of eye irritation for many classes of chemicals (e.g., gases, inorganic salts),
	•	poor generalization outside the original training domain,
	•	probability saturation effects (values clustering around 0.95).
	•	No domain applicability filter is implemented yet.
	•	The model has not undergone calibration, external validation, or regulatory-grade evaluation.

What this version is: 
	•	A working prototype of the full prediction pipeline.
	•	A demonstration of molecular featurization + ML inference + reporting.
	•	A robust software foundation for future improved models.

What this version is NOT:
	•	A validated toxicological classifier.
	•	A tool to support regulatory or safety decisions.
	•	A source of reliable irritation predictions for out-of-domain substances.



***Roadmap for the next releases***

Planned improvements for Version 0.2 / 0.3 include:
	•	Re-training the model on a diverse, balanced, and high-quality dataset
	•	Implementing a full domain of applicability (DoA) module
	•	Adding probability calibration (Platt scaling or isotonic regression)
	•	Benchmarking against public irritation datasets (e.g. ECHA, OECD)
	•	Implementing model diagnostics (SHAP, t-SNE, clustering)
	•	Providing interactive dashboards for prediction exploration
	•	Publishing performance metrics (ROC, PR curve, confusion matrix)


	## Installation

Clone the repository:

```bash
git clone https://github.com/AlinuxPal1/SDS-Eye_Irritation_Predict.git
cd SDS-Eye_Irritation_Predict

conda create -n sds_eye python=3.10
conda activate sds_eye

pip install -r requirements.txt

To run a full prediction on the example SDS file (data/test_sds_eye.csv), use:
python run_sds_eye_demo.py

Example output
Preview:
           Substance       SMILES  eye_smiles_valid  eye_irritation_prob  eye_irritation_flag
0  Hydrochloric acid           Cl              True             0.964762                  1.0
1      Sulfuric acid  OS(=O)(=O)O              True             0.744941                  1.0
2   Sodium hydroxide  [Na+].[OH-]              True             0.964762                  1.0
3        Acetic acid      CC(=O)O              True             0.944016                  1.0
4        Isopropanol       CC(C)O              True             0.951597                  1.0

Notes
	•	Invalid SMILES strings (e.g., ASD???) are flagged automatically.
	•	eye_irritation_flag uses a default threshold of 0.40 but can be customized.
