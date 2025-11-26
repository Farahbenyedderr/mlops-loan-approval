import joblib
import os
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.outliers import OutlierTrimmer


def prepare_data(df):
    """
    Clean, encode and preprocess the loan dataset before training.

    Args:
        df (pd.DataFrame): Raw dataset loaded from CSV.

    Raises:
        ValueError: If df is None or empty.
        KeyError: If required columns are missing.

    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe ready for ML.
    """

    # ====== Vérification des entrées ======
    if df is None or df.empty:
        raise ValueError("❌ Input dataframe is empty or None.")

    required_cols = [
        "person_age", "person_income", "person_emp_exp",
        "loan_amnt", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "loan_int_rate", "person_gender",
        "person_education", "person_home_ownership", "loan_intent",
        "previous_loan_defaults_on_file", "loan_status"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"❌ Missing required columns: {missing}")

    # ====== Prétraitement ======
    df["person_age"] = df["person_age"].astype(int)

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

    # ====== Gestion des outliers ======
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

    # ====== Vérification sortie ======
    if df_clean.empty:
        raise ValueError("❌ Cleaned dataframe is empty after preprocessing.")

    return df_clean


def train(df_clean):
    """
    Train an XGBoost classifier on the cleaned dataset.

    Args:
        df_clean (pd.DataFrame): Preprocessed dataframe returned by prepare_data().

    Raises:
        ValueError: If df_clean is empty or missing the 'loan_status' column.

    Returns:
        tuple:
            model (xgb.XGBClassifier): Trained model.
            x_test (pd.DataFrame): Test features.
            y_test (pd.Series): Test labels.
    """

    # ====== Vérification des entrées ======
    if df_clean is None or df_clean.empty:
        raise ValueError("❌ df_clean is empty or None. Cannot train model.")

    if "loan_status" not in df_clean.columns:
        raise KeyError("❌ 'loan_status' column is missing in df_clean.")

    # ====== Sélection des features corrélées ======
    threshold = 0.1
    corr = df_clean.corr()

    high_corr_features = corr.index[
        abs(corr["loan_status"]) > threshold
    ].tolist()

    # on retire loan_status de la liste
    if "loan_status" in high_corr_features:
        high_corr_features.remove("loan_status")

    if not high_corr_features:
        raise ValueError("❌ No correlated features found for training.")

    # ====== Définition X et y ======
    X = df_clean[high_corr_features]
    y = df_clean["loan_status"]

    # ====== Split train/test ======
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ====== Modèle XGBoost ======
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(x_train, y_train)

    # ====== Vérification sortie ======
    if model is None:
        raise ValueError("❌ Model training failed (model is None).")

    return model, x_test, y_test


def evaluate(model, x_test, y_test):
    """
    Evaluate a trained ML model using accuracy, AUC and classification report.

    Args:
        model: Trained machine learning model (must implement predict()).
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.

    Raises:
        ValueError: If model or test data are missing or empty.

    Returns:
        tuple:
            accuracy (float): Accuracy score.
            auc (float or None): AUC score (None if model does not support predict_proba).
            report (str): Classification report.
    """

    # ====== Vérifications ======
    if model is None:
        raise ValueError("❌ Model is None. Cannot evaluate.")

    if x_test is None or x_test.empty:
        raise ValueError("❌ x_test is empty or None.")

    if y_test is None or len(y_test) == 0:
        raise ValueError("❌ y_test is empty or None.")

    # ====== Predictions ======
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    # AUC (si predict_proba existe)
    try:
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    report = classification_report(y_test, y_pred)

    return acc, auc, report


def save(model, path="models/xgb_model.pkl"):
    """
    Save a trained model into the project-level models directory.
    Compatible with both:
        save(model)
        save(model, path)
        save(path, model)   <-- used in tests
    """

    # ✅ Gérer l'inversion des paramètres dans les tests
    if isinstance(model, str) and not isinstance(path, str):
        model, path = path, model

    if model is None:
        raise ValueError("❌ Cannot save model: model is None.")

    if not isinstance(path, str):
        raise TypeError("❌ Path must be a string.")

    # ✅ Racine du projet
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # ✅ Dossier models au niveau du projet
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # ✅ Nom du fichier uniquement
    full_path = os.path.join(model_dir, os.path.basename(path))

    joblib.dump(model, full_path)

    print(f"✅ Model réellement sauvegardé dans : {full_path}")
    return full_path


def load(path="models/xgb_model.pkl"):
    """
    Load a previously saved ML model from the project-level models directory.
    """

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_path = os.path.join(base_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ Model file not found: {full_path}")

    return joblib.load(full_path)


def load_metrics(path="models/svc_metrics.json"):
    """
    Load evaluation metrics from a JSON file.

    Args:
        path (str): Path to the metrics file.

    Returns:
        dict: Loaded metrics.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path, "r") as f:
        metrics = json.load(f)

    print(f"Metrics loaded: {path}")
    return metrics
