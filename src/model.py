import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.outliers import OutlierTrimmer


# --------------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------------
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

    required_columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 
        'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score',
        'loan_int_rate', 'person_education', 'person_gender', 
        'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file',
        'loan_status'
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colonnes manquantes: {missing_columns}")

    print("Nettoyage et prétraitement des données...")

    df = df.copy()

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


# --------------------------------------------------------
# TRAINING
# --------------------------------------------------------
def train_model(df, n_neighbors=5):
    """Train the KNN model."""
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    return model, X_test, y_test, X.columns.tolist()


# --------------------------------------------------------
# HYPERPARAMETER SEARCH
# --------------------------------------------------------
def find_optimal_k(df, k_range=range(1, 31)):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    k_scores = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        k_scores.append(score)

    best_k = k_range[np.argmax(k_scores)]
    best_score = max(k_scores)

    return best_k, best_score, k_scores


# --------------------------------------------------------
# EVALUATION
# --------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = None

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, auc, report, cm


# --------------------------------------------------------
# SAVE / LOAD
# --------------------------------------------------------
def save_model(obj, path):
    joblib.dump(obj, path)


def load_model(path):
    return joblib.load(path)


# --------------------------------------------------------
# PREDICT
# --------------------------------------------------------
def predict(model, data, feature_names):
    """Predict using pre-cleaned data (no preprocessing done here)."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame([data])

    df = df[feature_names]

    preds = model.predict(df)
    probs = model.predict_proba(df)

    return preds, probs
