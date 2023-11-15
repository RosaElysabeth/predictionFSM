import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import shap

def charger_modele():
    # Charger vos données
    data = pd.read_excel('etat_securite.xlsx')  # Assurez-vous d'avoir un fichier de données
    
    df = data.copy()
    
    # Prétraitement des données
    features = df.drop('Situation_Surpoids', axis=1)
    target = df['Situation_Surpoids']
    
    # Encodage
    code = {'Acceptable(Normale)': 1, 'Précaire': 2, 'Alarmante(Alerte)': 3, 'Critique(Urgence)': 4}  
    for col in df.select_dtypes('object').columns:
        df.loc[:, col] = df[col].replace(code)
    
    # Imputer les valeurs manquantes pour toutes les colonnes avec la stratégie 'most_frequent'
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Assurez-vous qu'il n'y a pas de NaN dans les données d'entraînement et de test
    assert not pd.isnull(df_imputed).any().any(), "Il y a des valeurs manquantes dans les données d'entraînement"
    
    # Division des données en ensembles d'entraînement et de test après gestion des NaN
    X_train, X_test, y_train, y_test = train_test_split(df_imputed, target, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle (exemple fictif)
    modele_securite_alimentaire = SVC(kernel='linear', C=1)
    modele_securite_alimentaire.fit(X_train, y_train)

    # Initialiser SHAP avec le modèle entraîné
    explainer = shap.Explainer(modele_securite_alimentaire, X_train)

    return modele_securite_alimentaire, explainer

def predire(modele, features):
    modele_securite_alimentaire, _ = modele
    prediction = modele_securite_alimentaire.predict(features)[0]
    return prediction

def calculer_shap(modele, features):
    _, explainer = modele

    # Assurez-vous que toutes les colonnes sont de type numérique
    features = features.apply(pd.to_numeric, errors='coerce')

    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(features)

    return shap_values
