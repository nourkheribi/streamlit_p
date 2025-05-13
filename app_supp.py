import streamlit as st
import pandas as pd
import joblib
import requests
import os
from sklearn.preprocessing import StandardScaler

# Fonction pour télécharger depuis Google Drive
def download_file_from_google_drive(share_url, output_filename):
    file_id = share_url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/file/d/1QOJXKCUxuNcpmIsZ_eEtUmyVqFpKdZD9/view?usp=sharing'
    response = requests.get(download_url)
    with open(output_filename, 'wb') as f:
        f.write(response.content)

# Télécharger le modèle depuis Drive (⚠️ Remplace le lien par celui du vrai .pkl)
drive_link = 'TON_LIEN_ICI'
model_path = 'rf2.pkl'
if not os.path.exists(model_path):
    download_file_from_google_drive(drive_link, model_path)

# Charger le modèle
model = joblib.load(model_path)

# Interface Streamlit
st.title("💊 Supplement Sales - Revenue Predictor")
st.write("Let's see how much 💸 your product makes!")

# Inputs
Category = st.selectbox('Product Category', ['Protein', 'Vitamin', 'Omega', 'Performance', 'Amino Acid', 'Mineral', 'Herbal', 'Sleep Aid', 'Fat Burner', 'Hydration'])
Units_Sold = st.number_input('Units Sold', min_value=0)
prix = st.number_input('Price ($)', min_value=0.0)
pourcentage_reduction = st.number_input('Discount Percentage (%)', min_value=0.0)
unites_retournees = st.number_input('Units Returned', min_value=0.0)
Location = st.selectbox('Location', ['Canada', 'UK', 'USA'])
platforme = st.selectbox('Platform', ['Amazon', 'Walmart', 'iHerb'])

# Mapping
category_map = {'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
                'Mineral': 5, 'Herbal': 6, 'Sleep Aid': 7, 'Fat Burner': 8, 'Hydration': 9}
location_map = {'Canada': 0, 'UK': 1, 'USA': 2}
platform_map = {'Amazon': 0, 'Walmart': 1, 'iHerb': 2}

# Data preparation
input_data = pd.DataFrame({
    'Category': [category_map[Category]],
    'Units Sold': [Units_Sold],
    'Price': [prix],
    'Discount': [pourcentage_reduction],
    'Units Returned': [unites_retournees],
    'Location': [location_map[Location]],
    'Platform': [platform_map[platforme]]
})

# Standardisation
scaler = StandardScaler()
input_data[['Units Sold', 'Price', 'Discount']] = scaler.fit_transform(input_data[['Units Sold', 'Price', 'Discount']])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    st.success(f'The predicted revenue is: ${prediction:.2f}')
