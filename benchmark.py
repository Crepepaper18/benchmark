import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection Benchmark",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'test_dataset' not in st.session_state:
    st.session_state.test_dataset = None
if 'train_dataset' not in st.session_state:
    st.session_state.train_dataset = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'selected_models' not in st.session_state:
    st.session_state.selected_models = []

# Define Google Drive path
gdrive_path = "/content/drive/MyDrive/train/"

# Main application logic
st.title("Benchmarking AI-Generated Fake News Detection Models")

st.markdown("""
## Welcome to the Fake News Detection Benchmark Tool

This application allows you to benchmark various transformer models on fake news detection tasks.
You can upload your own datasets, train multiple models, and compare their performance.

### Available Models:
- BERT
- RoBERTa
- DistilBERT
- XLNet

### Available Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

### Getting Started:
1. Navigate to the *Dataset Upload* page to upload your dataset
2. Use the *Model Training* page to train your models
3. View detailed evaluations on the *Model Evaluation* page
4. Compare models on the *Results Comparison* page

""")

# Create columns for displaying information
col1, col2 = st.columns(2)

with col1:
    st.header("Current Status")

    if st.session_state.dataset is not None:
        st.success(f"Dataset loaded: {len(st.session_state.dataset)} samples")
        if st.session_state.train_dataset is not None:
            st.info(f"Training set: {len(st.session_state.train_dataset)} samples")
        if st.session_state.test_dataset is not None:
            st.info(f"Test set: {len(st.session_state.test_dataset)} samples")
    else:
        st.warning("No dataset loaded")

    if st.session_state.model_results:
        st.success(f"Trained models: {len(st.session_state.model_results)}")
    else:
        st.warning("No models trained yet")

with col2:
    st.header("Quick Actions")

    if st.button("Load Sample Dataset"):
        with st.spinner("Loading sample dataset..."):
            try:
                train_path = os.path.join(gdrive_path, "train.tsv")
                test_path = os.path.join(gdrive_path, "test.tsv")
                val_path = os.path.join(gdrive_path, "validation.tsv")

                st.session_state.train_dataset = pd.read_csv(train_path, sep='\t')
                st.session_state.test_dataset = pd.read_csv(test_path, sep='\t')
                st.session_state.dataset = pd.concat([
                    st.session_state.train_dataset,
                    st.session_state.test_dataset
                ])
                st.success("Dataset loaded successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")

    if st.button("Clear All Data"):
        st.session_state.dataset = None
        st.session_state.train_dataset = None
        st.session_state.test_dataset = None
        st.session_state.model_results = {}
        st.session_state.selected_models = []
        st.success("All data cleared")
        st.experimental_rerun()