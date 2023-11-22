import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import shap

def charger_modele():
    # Chargement des données
    print("Chargement des données...")
    data = pd.read_excel('./etat_securite_test.xlsx')
    print("Données chargées avec succès.")
    
    df = data.copy()
    
    # Prétraitement des données
    features = df.drop('Situation_Surpoids', axis=1)
    target = df['Situation_Surpoids']
    
    # Encodage
    code = {'Acceptable(Normale)': 1, 'Précaire': 2, 'Alarmante(Alerte)': 3, 'Critique(Urgence)': 4}  
    for col in df.select_dtypes('object').columns:
        df.loc[:, col] = df[col].replace(code)
    
    # Imputation des valeurs manquantes pour toutes les colonnes avec la stratégie 'most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df_imputed, target, test_size=0.2, random_state=42)
    
    print(f"Debug - Forme des données : {df.shape}")
    print(f"Debug - Forme de X_train : {X_train.shape}, Forme de y_train : {y_train.shape}")

    # Création et entraînement du modèle
    print("Entraînement du modèle...")
    modele_securite_alimentaire = SVC(kernel='linear', C=1)
    modele_securite_alimentaire.fit(X_train, y_train)
    print("Modèle entraîné avec succès.")

    # Initialisation de SHAP avec le modèle entraîné
    print("Initialisation de SHAP...")
    explainer = shap.Explainer(modele_securite_alimentaire, X_train)
    print("SHAP initialisé avec succès.")

    return modele_securite_alimentaire, explainer, X_train, y_train

def predire(modele, features):
    modele_securite_alimentaire = modele
    prediction = modele_securite_alimentaire.predict(features)[0]
    return prediction

def calculer_shap(modele, features):
    _, explainer = modele
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Calcul des valeurs SHAP
    shap_values = explainer.shap_values(features)
    
    # print(shap_values)

    return shap_values

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

def interpret_shap(feature_names, shap_values, region_name):
    interpretations = []

    if shap_values.ndim == 1:
        for i, val in enumerate(shap_values):
            feature_name = feature_names[i]
            interpretations.append(
                f"Pour la région {region_name}, la valeur de la fonctionnalité '{feature_name}' a un impact de {val:.4f} sur la prédiction."
            )
    else:
        interpretations.append(
            f"Pour la région {region_name}, la valeur de la fonctionnalité a plusieurs composantes et ne peut pas être interprétée de manière simple. Les composantes sont : {shap_value.tolist()}"
        )

    return interpretations


def main():
    st.title("Prédiction de la Sécurité Alimentaire")

    # Charger le modèle
    modele, explainer, X_train, y_train = charger_modele()

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
            "Situation_Surpoids": [2],
            "Situation_MC": [1],
            "Situation_MA": [4],
        })
        
        # Utiliser les noms de colonnes de X_train pour garantir la cohérence
        features.columns = X_train.columns

        # Faire la prédiction
        prediction = predire(modele, features)

        # Afficher la prédiction
        st.write(f"Prédiction de la Classe: {prediction}")

        # Calculer et afficher les valeurs SHAP
        shap_values = calculer_shap((modele, explainer), features)

        # Obtenir l'ordre décroissant des indices des fonctionnalités par impact
        feature_order = list(reversed(np.argsort(shap_values[0])))

        print(f"Debug - Feature Order: {feature_order}")

        # Afficher les résultats SHAP avec des commentaires adaptés aux nutritionnistes
        st.write("Interprétation des Valeurs SHAP :")

        feature_names = features.columns  # Extraire les noms des fonctionnalités à l'extérieur de la boucle

        for feature_index in feature_order[0]:
            if 0 <= feature_index < len(shap_values[0]):
                feature_name = feature_names[feature_index]  # Utiliser le nom de la fonctionnalité ici
                shap_value = shap_values[0][feature_index]

                print(f"Debug - Feature Index: {feature_index}, Feature Name: {feature_name}, SHAP Value Shape: {shap_value.shape}")

                interpretation = interpret_shap(feature_names, shap_value, region_name)

                st.write(interpretation)

                print(f"Debug - All SHAP Values for {feature_name}:\n{shap_value}")

if __name__ == "__main__":
    main()
