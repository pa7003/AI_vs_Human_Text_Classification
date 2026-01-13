import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import joblib
from src.train_classifier import train_classifier, evaluate_classifier
from src.feature_extraction import extract_features
from src.data_utils import load_and_preprocess_data
from src.explainability import explain_prediction

st.title("AI vs Human Text Classifier")

@st.cache_resource
def load_model():
    # Placeholder: Load your trained model here
    return joblib.load('model.joblib')

@st.cache_resource
def load_tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

user_text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    tokenizer, model = load_tokenizer_model()
    inputs = tokenizer(user_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :].numpy()
    clf = load_model()
    pred = clf.predict(features)
    st.write(f"Prediction: {'Human' if pred[0]==0 else 'AI'}")
    # LIME explanation (placeholder)
    st.write("LIME explanation coming soon.")
