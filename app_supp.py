import streamlit as st
import zipfile
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Configure the page layout
st.set_page_config(page_title="Supplement Sales - Revenue Predictor", layout="centered")

# === 1. Model Loading Function ===
def charger_modele(zip_file):
    # Extract model from the uploaded ZIP
    with zipfile.ZipFile(zip_file, 'r') as zf:
        zf.extractall("extracted_files")  # Extract all contents to the folder 'extracted_files'
    
    # Assuming the model is named 'rf2.joblib' and is in the root of the ZIP
    model_path = "extracted_files/rf2.joblib"
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.sidebar.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            st.stop()  # Stop the app if loading the model fails
    else:
        st.sidebar.error("❌ Model not found in the ZIP file.")
        st.stop()

# === User Interface ===
st.title("💊 Supplement Sales - Revenue Predictor")
st.write("Upload a ZIP file containing your dataset and model to predict revenue.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload a ZIP file containing your dataset and model", type="zip")

# === Process the ZIP file ===
if uploaded_file is not None:
    # Load the model from the ZIP file
    modele = charger_modele(uploaded_file)

    # Extract the contents of the ZIP
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall("extracted_files")  # Extract to a folder named 'extracted_files'
    
    # List the extracted files to check for dataset
    extracted_files = os.listdir("extracted_files")
    st.write("Extracted files:", extracted_files)

    # Look for a CSV file in the extracted files
    csv_file = None
    for file in extracted_files:
        if file.endswith(".csv"):
            csv_file = file
            break

    if csv_file:
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(f"extracted_files/{csv_file}")
        st.write("Dataset preview:", data.head())

        # === Input Fields for Prediction ===
        Category = st.selectbox('Product Category', data['Category'].unique())
        Units_Sold = st.number_input('Units Sold', min_value=0)
        prix = st.number_input('Price ($)', min_value=0.0)
        pourcentage_reduction = st.number_input('Discount Percentage (%)', min_value=0.0)
        unites_retournees = st.number_input('Units Returned', min_value=0.0)
        Location = st.selectbox('Location', data['Location'].unique())
        platforme = st.selectbox('Platform', data['Platform'].unique())

        # === Mapping ===
        category_map = {'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
                        'Mineral': 5, 'Herbal': 6
