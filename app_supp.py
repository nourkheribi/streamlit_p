import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import gdown
from sklearn.preprocessing import StandardScaler

# ✅ Configure la page
st.set_page_config(page_title="Supplement Sales - Revenue Predictor", layout="centered")

# === 1. Chargement du modèle ===
def charger_modele():
    model_path = "models/rf2.joblib"  # Ensure this is the correct path
    os.makedirs("models", exist_ok=True)  # Create the 'models' folder if it doesn't exist

    if not os.path.exists(model_path):  # Check if model is already downloaded
        try:
            st.sidebar.warning("⚠ Téléchargement du modèle...")
            url = "https://drive.google.com/uc?id=1N5YXrUmStS3cmocrPcVs-WK7wFknEaRh"  # Correct direct URL for gdown
            gdown.download(url, model_path, quiet=False)
            st.sidebar.success("✅ Modèle téléchargé !")
        except Exception as e:
            st.sidebar.error(f"❌ Échec du téléchargement : {str(e)}")
            st.stop()  # Stop the app if there's an error during download

    try:
        # Load the model after download or if already present
        model = joblib.load(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erreur de chargement : {str(e)}")
        st.stop()  # Stop the app if loading the model fails

# === Charger le modèle ===
modele = charger_modele()

# === Interface Utilisateur ===
st.title("💊 Supplement Sales - Revenue Predictor")
st.write("Let's see how much 💸 your product makes!")

# === Inputs ===
Category = st.selectbox('Product Category', ['Protein', 'Vitamin', 'Omega', 'Performance', 'Amino Acid', 'Mineral', 'Herbal', 'Sleep Aid', 'Fat Burner', 'Hydration'])
Units_Sold = st.number_input('Units Sold', min_value=0)
prix = st.number_input('Price ($)', min_value=0.0)
pourcentage_reduction = st.number_input('Discount Percentage (%)', min_value=0.0)
unites_retournees = st.number_input('Units Returned', min_value=0.0)
Location = st.selectbox('Location', ['Canada', 'UK', 'USA'])
platforme = st.selectbox('Platform', ['Amazon', 'Walmart', 'iHerb'])

# === Mapping ===
category_map = {'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
                'Mineral': 5, 'Herbal': 6, 'Sleep Aid': 7, 'Fat Burner': 8, 'Hydration': 9}
location_map = {'Canada': 0, 'UK': 1, 'USA': 2}
platform_map = {'Amazon': 0, 'Walmart': 1, 'iHerb': 2}

# === DataFrame ===
input_data = pd.DataFrame({
    'Category': [category_map[Category]],
    'Units Sold': [Units_Sold],
    'Price': [prix],
    'Discount': [pourcentage_reduction],
    'Units Returned': [unites_retournees],
    'Location': [location_map[Location]],
    'Platform': [platform_map[platforme]]
})

# === Standardisation ===
scaler = StandardScaler()
input_data[['Units Sold', 'Price', 'Discount']] = scaler.fit_transform(input_data[['Units Sold', 'Price', 'Discount']])

# === Prédiction ===
if st.button('Predict'):
    prediction = modele.predict(input_data)[0]
    st.success(f'The predicted revenue is: ${prediction:.2f}')
