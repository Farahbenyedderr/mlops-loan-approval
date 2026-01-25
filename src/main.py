import argparse
import json

import joblib
import mlflow.sklearn
import pandas as pd

import mlflow

try:
    # Try absolute import first (works when running main.py directly)
    from model import evaluate, input_preperation, predict, prepare_data, save, train
except ImportError:
    # Fallback for package execution
    import sys
    import os
    # Add parent folder to sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import evaluate, input_preperation, predict, prepare_data, save, train
mlflow.set_tracking_uri("http://127.0.0.1:5000")

def predict_main(input_data, model_path="models/trained_model.pkl"):
    """Make predictions using the trained model."""
    model = joblib.load(model_path)
    print(input_data)
    input_df = input_preperation(input_data)
    print("output:", input_df)
    predictions = predict(model, input_df)
    return predictions


def prepare_data_main(file_path="data/loan_data.csv"):
    df = pd.read_csv(file_path)

    clean_data, training_data, testing_data = prepare_data(df)

    clean_data.to_csv("data/cleaned_loan_data.csv", index=False)
    training_data.to_csv("data/train_data.csv", index=False)
    testing_data.to_csv("data/test_data.csv", index=False)


def train_model_main(params=None):
    # ... load data ...
    training_data = pd.read_csv("data/train_data.csv")
    testing_data = pd.read_csv("data/test_data.csv")

    # Start the MLflow Run here
    mlflow.set_experiment("Loan_Prediction_ExperimentV2")

    with mlflow.start_run():
        # 1. Train (Logs params automatically now)
        model = train(training_data, params=params, use_mlflow=True)

        # 2. Evaluate (Logs metrics automatically now)
        evaluate(model, testing_data, use_mlflow=True)

        # 3. Save (Logs model artifact automatically now)
        save(model, path="models/trained_model.pkl", use_mlflow=True)


def evaluate_model_main(
    model_path="models/trained_model.pkl", test_data_path="data/test_data.csv"
):

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    acc, auc, report = evaluate(model, test_df)
    print("Model Evaluation:")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)
    return acc, auc, report


def end_to_end_main():
    prepare_data_main()
    train_model_main()
    evaluate_model_main()
    return 1


# ... ensure your prepare/train/evaluate functions are imported or defined here ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_data_main", action="store_true")
    parser.add_argument("--train_model_main", action="store_true")
    parser.add_argument("--evaluate_model_main", action="store_true")

    # Accept params as a JSON string
    parser.add_argument("--params", type=str, default=None)

    args = parser.parse_args()

    # 1. Parse the JSON string into a Python Dictionary
    params_dict = None
    if args.params:
        try:
            params_dict = json.loads(args.params)
            print(f"Received custom parameters: {params_dict}")
        except json.JSONDecodeError as e:
            print(f"Error parsing params JSON: {e}")
            print("Using default parameters.")
            params_dict = None

    # 2. Execute logic
    if args.prepare_data_main:
        prepare_data_main()

    elif args.train_model_main:
        # Update: Pass the dictionary here
        train_model_main(params=params_dict)

    elif args.evaluate_model_main:
        evaluate_model_main()

    else:
        # Default pipeline (Run All)
        print("Running default pipeline...")
        prepare_data_main()

        # Update: Pass the dictionary here as well
        train_model_main(params=params_dict)

        evaluate_model_main()


if __name__ == "__main__":
    main()
