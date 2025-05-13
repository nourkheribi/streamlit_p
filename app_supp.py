import streamlit as st
import joblib
import os
import zipfile
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ✅ Configure the page layout
st.set_page_config(page_title="Supplement Sales - Revenue Predictor", layout="centered")

# === Model Loading Function ===
def charger_modele_depuis_zip():
    zip_path = "model_rf2.zip"
    model_path = "models/rf2.joblib"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        if not os.path.exists(zip_path):
            st.sidebar.error("❌ Le fichier 'model_rf2.zip' est manquant.")
            st.stop()

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")
            st.sidebar.success("✅ Modèle extrait avec succès")
        except Exception as e:
            st.sidebar.error(f"❌ Erreur lors de l'extraction : {str(e)}")
            st.stop()

    try:
        modele = joblib.load(model_path)
        st.sidebar.success("✅ Modèle chargé avec succès")
        return modele
    except Exception as e:
        st.sidebar.error(f"❌ Échec du chargement du modèle : {str(e)}")
        st.stop()

# === Load the model ===
modele = charger_modele_depuis_zip()

# === User Interface ===
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
category_map = {
    'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
    'Mineral': 5, 'Herbal': 6, 'Sleep Aid': 7, 'Fat Burner': 8, 'Hydration': 9
}
location_map = {'Canada': 0, 'UK': 1, 'USA': 2}
platform_map = {'Amazon': 0, 'Walmart': 1, 'iHerb': 2}

# === Create DataFrame ===
input_data = pd.DataFrame({
    'Category': [category_map[Category]],
    'Units Sold': [Units_Sold],
    'Price': [prix],
    'Discount': [pourcentage_reduction],
    'Units Returned': [unites_retournees],
    'Location': [location_map[Location]],
    'Platform': [platform_map[platforme]]
})

# === Standardization ===
scaler = StandardScaler()
input_data[['Units Sold', 'Price', 'Discount']] = scaler.fit_transform(
    input_data[['Units Sold', 'Price', 'Discount']]
)

# === Prediction ===
if st.button('Predict'):
    prediction = modele.predict(input_data)[0]
    st.success(f'The predicted revenue is: ${prediction:.2f}')
