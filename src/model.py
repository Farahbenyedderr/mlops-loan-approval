import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from feature_engine.outliers import OutlierTrimmer
from RF_model import train_random_forest


def prepare_data(df):
    """Clean and encode dataset before training."""

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

    df = df.assign(
    person_education=df["person_education"].replace({
        "High School": 0,
        "Associate": 1,
        "Bachelor": 2,
        "Master": 3,
        "Doctorate": 4,
    })
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
    
    df_trimmed = trimmer.fit_transform(df)
    threshold = 0.1
    corr = df_trimmed.corr()

    high_corr_features = corr.index[
        abs(corr["loan_status"]) > threshold
    ].tolist()

    clean_data =  df_trimmed[high_corr_features]
    
    X = clean_data.drop("loan_status", axis=1)
    y = clean_data["loan_status"]    
    

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    training_data = pd.concat([x_train, y_train], axis=1)
    testing_data = pd.concat([x_test, y_test], axis=1)
    
    return clean_data, training_data, testing_data
    
    


def train(train_df):
    
    x_train = train_df.drop("loan_status", axis=1)
    y_train = train_df["loan_status"]
    
    """"Train Random Forest Model"""
    model = train_random_forest(x_train, y_train)
    
    """Train XGBoost classifier."""
    """ model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(x_train, y_train) """

    return model


def evaluate(model, test_df):
    """Evaluate model accuracy and AUC."""

    x_test = test_df.drop("loan_status", axis=1)
    y_test = test_df["loan_status"]
    
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)

    try:
        y_prob = model.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = None

    report = classification_report(y_test, y_pred)

    return acc, auc, report


def save(model, path="models/xgb_model.pkl"):
    """Save trained model to file."""
    joblib.dump(model, path)


def load(path="models/xgb_model.pkl"):
    """Load trained model from file."""
    return joblib.load(path)
