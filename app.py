import streamlit as st
import pandas as pd
import numpy as np 
from modele import charger_modele, predire, calculer_shap

# Fonction pour mapper les noms de région à des valeurs numériques
def map_region_to_numeric(region_name):
    region_mapping = {
        "madagascar": 0,
        "diana": 1, 
        "sava": 2,
        "itasy": 3,
        "analamanga": 4,
        "vakinankaratra": 5,
        "bongolava": 6,
        "sofia": 7,
        "boeny": 8,
        "betsiboka": 9,
        "melaky": 10,
        "alaotra-mangoro": 11,
        "antsinana": 12,
        "analanjirofo": 13,
        "ambatosoa": 14,
        "amoron'i mania": 15,
        "vatovavy": 16,
        "fitovinany": 17,
        "haute matsiatra": 18,
        "atsimo-atsinanana": 19,
        "ihorombe": 20,
        "menabe": 21,
        "atsimo-andrefana": 22,
        "androy": 23
    }
    
    # Convertir la région en minuscules avant de chercher dans le dictionnaire
    region_name_lower = region_name.lower()
    
    # Retourne la valeur numérique correspondante si elle existe, sinon retourne -1 ou une valeur par défaut
    return region_mapping.get(region_name_lower, -1)


@st.cache_data
def interpret_shap(feature_name, shap_value, region_name):
    """
    Interprète une valeur SHAP pour une fonctionnalité donnée.
    """
    interpretations = []

    if shap_value.ndim == 1:
        # Si la valeur SHAP est un tableau unidimensionnel
        for i, val in enumerate(shap_value):
            interpretations.append(
                f"Pour la région {region_name}, la valeur de la fonctionnalité '{feature_name}' a un impact de {val:.4f} sur la prédiction."
            )
    else:
        # Si la valeur SHAP est un tableau multidimensionnel
        interpretations.append(
            f"Pour la région {region_name}, la valeur de la fonctionnalité '{feature_name}' a plusieurs composantes et ne peut pas être interprétée de manière simple. Les composantes sont : {shap_value.tolist()}"
        )

    return interpretations

def main():
    st.title("Prédiction de la Sécurité Alimentaire")

    # Charger le modèle
    modele = charger_modele()

    # Interface utilisateur pour saisir les fonctionnalités
    region_name = st.text_input("Entrez la région:")
    region_numeric = map_region_to_numeric(region_name)

    # Vérifier si la région est valide
    if region_numeric == -1:
        st.warning("La région saisie n'est pas valide. Veuillez saisir une région valide.")
        return

    date = st.date_input("Entrez la date:")
    
    # Extract features from date
    year = date.year
    month = date.month
    day = date.day

    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        # Prétraitement des fonctionnalités (excluant "Situation_Surpoids")
        features = pd.DataFrame({
            "DATE": [year],
            "REGION": [region_numeric],
            "Situation_Surpoids":[0],
            "Situation_MC": [0], 
            "Situation_MA": [0]
        }) 

        # Faire la prédiction
        prediction = predire(modele, features)

        # Afficher la prédiction
        st.write(f"Prédiction de la Classe: {prediction}")

        # Calculer et afficher les valeurs SHAP
        shap_values = calculer_shap(modele, features)
        
        # Obtenir l'ordre décroissant des indices des fonctionnalités par impact
        feature_order = list(reversed(np.argsort(shap_values[0])))

        print(f"Debug - Feature Order: {feature_order}")
  
  
        # Afficher les résultats SHAP avec des commentaires adaptés aux nutritionnistes
        st.write("Interprétation des Valeurs SHAP :")
        for feature_index in feature_order[0]:
            if 0 <= feature_index < len(shap_values[0]):
                feature_name = features.columns[feature_index]
                shap_value = shap_values[0][feature_index]

                print(f"Debug - Feature Index: {feature_index}, Feature Name: {feature_name}, SHAP Value Shape: {shap_value.shape}")

                interpretation = interpret_shap(feature_name, shap_value, region_name)

                st.write(interpretation)


        print(f"Debug - All SHAP Values for {feature_name}:\n{shap_value}")



if __name__ == "__main__":
    main()
