import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from feature_engine.outliers import OutlierTrimmer
import joblib
import os
import json


def prepare_data(df):
    """
    Charge et prétraite les données pour l'entraînement du modèle.
    
    Args:
        df (pd.DataFrame): DataFrame brut contenant les données de prêt
        
    Returns:
        pd.DataFrame: DataFrame nettoyé et prétraité
        
    Raises:
        ValueError: Si le DataFrame est vide ou si des colonnes requises sont manquantes
    """
    # Vérification des entrées
    if df.empty:
        raise ValueError("Le DataFrame fourni est vide")
    
    required_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
                       'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
                       'loan_int_rate', 'person_education', 'person_gender', 
                       'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file',
                       'loan_status']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")
    
    print("Nettoyage et prétraitement des données...")
    
    # Conversion des types
    df["person_age"] = df["person_age"].astype(int)

    # Normalisation des colonnes numériques
    skewed_cols = [
        "person_age", "person_income", "person_emp_exp",
        "loan_amnt", "loan_percent_income",
        "cb_person_cred_hist_length", "credit_score"
    ]

    norm_cols = ["loan_int_rate"]

    scaler = StandardScaler()
    normalizer = MinMaxScaler()

    df[skewed_cols] = scaler.fit_transform(df[skewed_cols])
    df[norm_cols] = normalizer.fit_transform(df[norm_cols])

    # Encodage des variables catégorielles
    df["person_education"].replace(
        {
            "High School": 0,
            "Associate": 1,
            "Bachelor": 2,
            "Master": 3,
            "Doctorate": 4,
        },
        inplace=True,
    )

    df["person_gender"] = df["person_gender"].map({"male": 0, "female": 1})

    df["person_home_ownership"] = df["person_home_ownership"].map(
        {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    )

    df["loan_intent"] = df["loan_intent"].map(
        {
            "PERSONAL": 0,
            "EDUCATION": 1,
            "MEDICAL": 2,
            "VENTURE": 3,
            "HOMEIMPROVEMENT": 4,
            "DEBTCONSOLIDATION": 5,
        }
    )

    df["previous_loan_defaults_on_file"] = df[
        "previous_loan_defaults_on_file"
    ].map({"No": 0, "Yes": 1})

    # Gestion des outliers
    trimmer = OutlierTrimmer(
        capping_method="iqr",
        tail="right",
        variables=[
            "person_age", "person_gender", "person_education",
            "person_income", "person_emp_exp",
            "person_home_ownership", "loan_amnt",
            "loan_intent", "loan_int_rate", "loan_percent_income",
            "cb_person_cred_hist_length", "credit_score",
            "previous_loan_defaults_on_file",
        ],
    )

    df_clean = trimmer.fit_transform(df)
    print(f"Données prétraitées: {df_clean.shape[0]} lignes, {df_clean.shape[1]} colonnes")
    return df_clean


def train_model(df_clean, test_size=0.2, random_state=42):
    """
    Entraîne un modèle SVC sur les données préparées.
    
    Args:
        df_clean (pd.DataFrame): DataFrame nettoyé et prétraité
        test_size (float): Proportion des données pour le test (default: 0.2)
        random_state (int): Seed pour la reproductibilité (default: 42)
        
    Returns:
        tuple: (model, X_test, y_test, X_train, y_train)
            - model: Modèle SVC entraîné
            - X_test: Features de test
            - y_test: Labels de test
            - X_train: Features d'entraînement
            - y_train: Labels d'entraînement
            
    Raises:
        ValueError: Si 'loan_status' n'est pas dans les colonnes
    """
    # Vérification des entrées
    if 'loan_status' not in df_clean.columns:
        raise ValueError("La colonne 'loan_status' est requise pour l'entraînement")
    
    print("Entraînement du modèle SVC...")
    
    # Configuration X et y
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status']
    
    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"   Données d'entraînement: {X_train.shape[0]} échantillons")
    print(f"   Données de test: {X_test.shape[0]} échantillons")
    
    # Initialisation du modèle SVC
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=random_state
    )
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    print("Modèle entraîné avec succès")
    
    return model, X_test, y_test, X_train, y_train


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle entraîné.
    
    Args:
        model: Modèle entraîné
        X_test (pd.DataFrame/array): Features de test
        y_test (pd.Series/array): Labels de test
        
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
            - accuracy: Précision globale
            - auc_score: Score AUC-ROC
            - classification_report: Rapport de classification détaillé
            - confusion_matrix: Matrice de confusion
            - roc_curve: Données pour la courbe ROC
    """
    print("Évaluation des performances du modèle...")
    
    # Vérification des entrées
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("Les données de test ne peuvent pas être vides")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Rapport de classification
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Métriques retournées
    metrics = {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }
    
    print(f"Évaluation terminée - Précision: {accuracy:.2%}")
    return metrics


def save_model(model, path="models/svc_model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.
    
    Args:
        model: Modèle entraîné à sauvegarder
        path (str): Chemin où sauvegarder le modèle
        
    Returns:
        str: Chemin où le modèle a été sauvegardé
    """
    # Création du dossier si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarde du modèle
    joblib.dump(model, path)
    print(f"Modèle sauvegardé: {path}")
    
    return path


def load_model(path="models/svc_model.pkl"):
    """
    Charge un modèle sauvegardé depuis un fichier.
    
    Args:
        path (str): Chemin vers le modèle sauvegardé
        
    Returns:
        model: Modèle chargé
        
    Raises:
        FileNotFoundError: Si le fichier modèle n'existe pas
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier modèle non trouvé: {path}")
    
    model = joblib.load(path)
    print(f"Modèle chargé: {path}")
    
    return model


def save_metrics(metrics, path="models/svc_metrics.json"):
    """
    Sauvegarde les métriques d'évaluation dans un fichier JSON.
    
    Args:
        metrics (dict): Dictionnaire des métriques à sauvegarder
        path (str): Chemin où sauvegarder les métriques
    """
    # Création du dossier si nécessaire
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Sauvegarde des métriques
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Métriques sauvegardées: {path}")


def load_metrics(path="models/svc_metrics.json"):
    """
    Charge les métriques d'évaluation depuis un fichier JSON.
    
    Args:
        path (str): Chemin vers le fichier de métriques
        
    Returns:
        dict: Métriques chargées
        
    Raises:
        FileNotFoundError: Si le fichier de métriques n'existe pas
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de métriques non trouvé: {path}")
    
    with open(path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Métriques chargées: {path}")
    return metrics