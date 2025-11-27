import pandas as pd
import numpy as np

from model import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def test_prepare_data():
    print("🧪 Testing prepare_data()...")

    sample = {
        'person_age': [25, 30, 35],
        'person_gender': ['male', 'female', 'male'],
        'person_education': ['Bachelor', 'Master', 'High School'],
        'person_income': [50000, 75000, 40000],
        'person_emp_exp': [2, 5, 1],
        'person_home_ownership': ['RENT', 'MORTGAGE', 'OWN'],
        'loan_amnt': [10000, 15000, 8000],
        'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL'],
        'loan_int_rate': [10.5, 12.0, 8.5],
        'loan_percent_income': [0.2, 0.15, 0.25],
        'cb_person_cred_hist_length': [3, 5, 2],
        'credit_score': [650, 700, 600],
        'previous_loan_defaults_on_file': ['No', 'Yes', 'No'],
        'loan_status': [1, 0, 1]
    }

    df = pd.DataFrame(sample)
    df_clean = prepare_data(df)

    print(df_clean.head())
    print("Prepare data OK ✔")

    return df_clean


def test_train_model(df_clean):
    print("\n🧪 Testing train_model()...")

    model, X_test, y_test, feature_names = train_model(df_clean)

    print("Model trained ✔")
    print("Features:", feature_names)

    return model, X_test, y_test, feature_names


def test_evaluate_model(model, X_test, y_test):
    print("\n🧪 Testing evaluate_model()...")

    acc, auc, report, cm = evaluate_model(model, X_test, y_test)

    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)
    print(cm)

    print("Evaluation OK ✔")
    return acc, auc, report, cm


def test_save_load_model(model, feature_names):
    print("\n🧪 Testing save/load...")

    save_model((model, feature_names), "test_model.pkl")
    loaded_model, loaded_feature_names = load_model("test_model.pkl")

    assert model.n_neighbors == loaded_model.n_neighbors

    print("Save/load OK ✔")
    return loaded_model, loaded_feature_names


def test_predictions(model, X_test):
    print("\n🧪 Testing predictions...")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("Predictions:", preds[:5])
    print("Probs:", probs[:5])

    print("Prediction OK ✔")


def run_all_tests():
    print("🚀 Running all tests...\n")

    df_clean = test_prepare_data()
    model, X_test, y_test, feature_names = test_train_model(df_clean)
    accuracy, auc, report, cm = test_evaluate_model(model, X_test, y_test)

    loaded_model, loaded_features = test_save_load_model(model, feature_names)
    test_predictions(loaded_model, X_test)

    print("\n🎉 All tests passed successfully!")


if __name__ == "__main__":
    run_all_tests()
