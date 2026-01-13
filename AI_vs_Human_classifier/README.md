# AI_vs_Human_classifier

This project detects whether a given text was written by an AI or a human. It includes dataset handling, transformer-based feature extraction, classifier training, evaluation, LIME explainability, and a Streamlit web app for live predictions.

## Features
- Load and preprocess dataset with AI and human text samples
- Feature extraction using BERT or RoBERTa
- Classifier training and evaluation (accuracy, F1, ROC-AUC)
- LIME for explainability
- Streamlit app for live predictions
- Jupyter notebook with full workflow

## Structure
- `src/`: Source code for data processing, modeling, and explainability
- `notebooks/`: Jupyter notebooks for workflow demonstration
- `streamlit_app/`: Streamlit web app code
- `data/`: Place your datasets here

## Quick Start
1. Install requirements: `pip install -r requirements.txt`
2. Run the notebook in `notebooks/`
3. Launch the Streamlit app: `streamlit run streamlit_app/app.py`
