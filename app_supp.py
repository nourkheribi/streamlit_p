import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Charger le modèle
model = joblib.load('rf2.pkl')

# Créer un titre pour l'application
st.title("Supplement_Sales - Revenue Predictor")
st.write('What is the revenue of a supplement product?')

# Demander à l'utilisateur d'entrer des valeurs pour les colonnes
Category = st.selectbox('Product Category', ['Protein', 'Vitamin', 'Omega', 'Performance', 'Amino Acid', 'Mineral', 'Herbal', 'Sleep Aid', 'Fat Burner', 'Hydration'])
Units_Sold = st.number_input('Units Sold', min_value=0)
prix = st.number_input('Price', min_value=0.0)
pourcentage_reduction = st.number_input('Discount Percentage', min_value=0.0)
unités_Retournés = st.number_input('Units Returned', min_value=0.0)
Location = st.selectbox('Location', ['Canada', 'UK', 'USA'])
platforme = st.selectbox('Platform', ['Amazon', 'Walmart', 'iHerb'])

# Encodeur pour les colonnes catégorielles
label_encoder = LabelEncoder()

# Dictionnaire pour les catégories codées
category_map = {'Protein': 0, 'Vitamin': 1, 'Omega': 2, 'Performance': 3, 'Amino Acid': 4,
                'Mineral': 5, 'Herbal': 6, 'Sleep Aid': 7, 'Fat Burner': 8, 'Hydration': 9}

location_map = {'Canada': 0, 'UK': 1, 'USA': 2}
platform_map = {'Amazon': 0, 'Walmart': 1, 'iHerb': 2}

# Préparer les données d'entrée
input_data = {
    'Category': [category_map[Category]],
    'Units Sold': [Units_Sold],
    'Price': [prix],
    'Discount': [pourcentage_reduction],
    'Units Returned': [unités_Retournés],
    'Location': [location_map[Location]],
    'Platform': [platform_map[platforme]]
}

# Convertir les données en DataFrame
input_df = pd.DataFrame(input_data)

# Normalisation des colonnes numériques
scaler = StandardScaler()
input_df[['Units Sold', 'Price', 'Discount']] = scaler.fit_transform(input_df[['Units Sold', 'Price', 'Discount']])

# Afficher les données d'entrée pour vérifier
#st.write(input_df)

# Prédiction lorsque l'utilisateur appuie sur le bouton "Predict"
if st.button('Predict'):
    # Prédire avec le modèle
    prediction = model.predict(input_df)[0]
    st.write(f'The predicted revenue is: {prediction:.2f}')
