import joblib
import mlflow.sklearn
import pandas as pd
import skops.io as sio
from feature_engine.outliers import OutlierTrimmer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import mlflow
try:
    # Try absolute import first (works when running main.py directly)
    from es_logger import log_to_es
except ImportError:
    # Fallback for package execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from es_logger import log_to_es

# Handle import relative to where the script is run
try:
    from .RF_model import train_random_forest
except ImportError:
    from RF_model import train_random_forest


mlflow.set_tracking_uri("http://127.0.0.1:5000")



def predict(model, input_data):
    """Make predictions using the trained model."""
    return model.predict(input_data)


def input_preperation(input_data):
    """Prepares single input for prediction using saved scalers."""
    df = pd.DataFrame([input_data.dict()])

    skewed_cols = ["person_income", "loan_amnt", "loan_percent_income"]
    norm_cols = ["loan_int_rate"]

    # Load scalers securely
    with open("models/scaler.pkl", "rb") as f:
        scaler = sio.load(f, trusted=True)

    with open("models/normalizer.pkl", "rb") as f:
        normalizer = sio.load(f, trusted=True)

    df[skewed_cols] = scaler.transform(df[skewed_cols])
    df[norm_cols] = normalizer.transform(df[norm_cols])

    df["person_home_ownership"] = df["person_home_ownership"].map(
        {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    )
    df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(
        {"N": 0, "Y": 1}
    )
    return df


def prepare_data(df):
    """Clean, encode, and split dataset before training."""
    print("Preparing data...")
    df["person_age"] = df["person_age"].astype(int)

    skewed_cols = [
        "person_age",
        "person_emp_exp",
        "cb_person_cred_hist_length",
        "credit_score",
    ]
    skewed_cols2 = ["person_income", "loan_amnt", "loan_percent_income"]
    norm_cols = ["loan_int_rate"]

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    normalizer = MinMaxScaler()

    df[skewed_cols] = scaler1.fit_transform(df[skewed_cols])
    df[skewed_cols2] = scaler2.fit_transform(df[skewed_cols2])
    df[norm_cols] = normalizer.fit_transform(df[norm_cols])

    # Save scalers securely
    with open("models/scaler.pkl", "wb") as f:
        sio.dump(scaler2, f)
    with open("models/normalizer.pkl", "wb") as f:
        sio.dump(normalizer, f)
    print("✓ Scalers saved to models/")

    df = df.assign(
        person_education=df["person_education"].replace(
            {
                "High School": 0,
                "Associate": 1,
                "Bachelor": 2,
                "Master": 3,
                "Doctorate": 4,
            }
        )
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
    df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(
        {"No": 0, "Yes": 1}
    )

    trimmer = OutlierTrimmer(
        capping_method="iqr",
        tail="right",
        variables=[
            "person_age",
            "person_gender",
            "person_education",
            "person_income",
            "person_emp_exp",
            "person_home_ownership",
            "loan_amnt",
            "loan_intent",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "previous_loan_defaults_on_file",
        ],
    )
    df_trimmed = trimmer.fit_transform(df)

    threshold = 0.1
    corr = df_trimmed.corr()
    high_corr_features = corr.index[abs(corr["loan_status"]) > threshold].tolist()
    if "loan_amnt" not in high_corr_features:
        high_corr_features.append("loan_amnt")

    print("High correlation features:", high_corr_features)
    clean_data = df_trimmed[high_corr_features]

    X = clean_data.drop("loan_status", axis=1)
    y = clean_data["loan_status"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    training_data = pd.concat([x_train, y_train], axis=1)
    testing_data = pd.concat([x_test, y_test], axis=1)

    return clean_data, training_data, testing_data


def train(train_df, params=None, use_mlflow=True):
    """
    Trains the Random Forest model and logs hyperparameters to MLflow.
    """
    # Separate features and target
    x_train = train_df.drop("loan_status", axis=1)
    y_train = train_df["loan_status"]

    # Train Random Forest Model
    model = train_random_forest(x_train, y_train, params=params)

    # --- MLflow Logging: Hyperparameters ---
    if use_mlflow:
        print("   Logging params to MLflow...")
        # Get all parameters actually used by the model
        all_params = model.get_params()

        # If user passed specific params, we can log specifically those,
        # but logging all_params is safer for reproducibility.
        mlflow.log_params(all_params)
        log_to_es("mlflow_params", {"params": all_params})

    return model


def evaluate(model, test_df, use_mlflow=True):
    """
    Evaluate model accuracy and AUC, and log metrics to MLflow.
    """
    x_test = test_df.drop("loan_status", axis=1)
    y_test = test_df["loan_status"]

    y_pred = model.predict(x_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    report = classification_report(y_test, y_pred)

    # --- MLflow Logging: Metrics ---

    if use_mlflow:
        print("   Logging metrics to MLflow...")
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc if auc else 0.0,
        }
        mlflow.log_metrics(metrics)
        log_to_es("mlflow_metrics", metrics)

    return acc, auc, report


def save(model, path="models/xgb_model.pkl", use_mlflow=True):
    """
    Save trained model locally AND to MLflow artifacts.
    """
    # 1. Save locally (for API usage)
    joblib.dump(model, path)
    print(f"   Model saved locally to {path}")

    # 2. Save to MLflow (for versioning/registry)
    if use_mlflow:
        print("   Logging model artifact to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name="LoanDefaultModel_RF",  # Optional: Registers model directly
        )
        log_to_es("mlflow_models", {"path": path})


def load(path="models/xgb_model.pkl"):
    """Load trained model from file."""
    return joblib.load(path)
