import streamlit as st
import pandas as pd
import numpy as np 
from modele import charger_modele, predire, calculer_shap

def generate_default_interpretation(feature):
    # Fonction pour générer une interprétation par défaut pour une fonctionnalité donnée
    return f"La colonne {feature} a un impact nutritionnel. Le coefficient associé est {{:.6f}}."

def interpret_shap(feature, shap_value, region):
    # Fonction pour interpréter les valeurs SHAP en fonction de la région
    interpretation = ""

    # Dictionnaire des interprétations pour chaque fonctionnalité et région
    feature_interpretations = {}

    # Générer des interprétations par défaut pour chaque fonctionnalité
    features = ["DATE", "REGION", "Situation_Surpoids", "Situation_MC", "Situation_MA"]
    for feature in features:
        feature_interpretations[feature] = {
            "default": generate_default_interpretation(feature)
        }

    # Ajoutez d'autres fonctionnalités au besoin

    # Vérifier si la fonctionnalité a une interprétation spécifique pour la région
    if region in feature_interpretations.get(feature, {}):
        interpretation = feature_interpretations[feature][region]
    else:
        # Sinon, utiliser l'interprétation par défaut
        interpretation = feature_interpretations[feature].get("default", f"Interprétation non définie pour la fonctionnalité {feature}.")

    # Si la fonctionnalité a un coefficient, remplacez la partie {:.6f} par la valeur réelle
    if "{:.6f}" in interpretation:
        interpretation = interpretation.format(np.array2string(shap_value, precision=6, separator=', ', suppress_small=True))


    return interpretation

def main():
    st.title("Prédiction de la Sécurité Alimentaire")

    # Charger le modèle
    modele = charger_modele()

    # Interface utilisateur pour saisir les fonctionnalités
    region = st.text_input("Entrez la région:")
    date = st.date_input("Entrez la date:")
    
    # Extract features from date
    year = date.year
    month = date.month
    day = date.day

    # Bouton pour effectuer la prédiction
    if st.button("Prédire"):
        # Prétraitement des fonctionnalités
        features = pd.DataFrame({"DATE": [year], "REGION": [region], "Situation_Surpoids": [0], "Situation_MC": [0], "Situation_MA": [0]}) 
    
        # Faire la prédiction
        prediction = predire(modele, features)

        # Afficher la prédiction
        st.write(f"Prédiction de la Classe: {prediction}")

        # Calculer et afficher les valeurs SHAP
        shap_values = calculer_shap(modele, features)

        # Obtenir l'ordre décroissant des indices des fonctionnalités par impact
        feature_order = list(reversed(np.argsort(shap_values[0])[:]))

        # Afficher les résultats SHAP avec des commentaires adaptés aux nutritionnistes
        st.write("Interprétation des Valeurs SHAP :")
        for feature_index in feature_order:
            if (0 <= feature_index).all() and (feature_index < len(shap_values[0])).all():
                feature = features.columns[feature_index]
                shap_value = shap_values[0][feature_index]
                interpretation = interpret_shap(feature, shap_value, region)
                st.write(interpretation)



if __name__ == "__main__":
    main()
