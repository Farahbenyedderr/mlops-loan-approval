import argparse
import pandas as pd
from model import (
    prepare_data,
    train,
    evaluate,
    save,
    load,
)


def train_pipeline():
    """Execute the complete training pipeline."""
    print("\n===== TRAINING PIPELINE STARTED =====")

    # Load raw data
    df = pd.read_csv("data/loan_data.csv")
    print("✔ Data loaded successfully")

    # Prepare data
    df_clean = prepare_data(df)
    print("✔ Data preparation completed")

    # Train model
    model, X_test, y_test = train(df_clean)
    print("✔ Model trained successfully")

    # Evaluate model
    acc, auc, report = evaluate(model, X_test, y_test)
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(report)

    # Save trained model
    save(model)
    print("✔ Model saved to models/xgb_model.pkl")

    print("===== TRAINING PIPELINE FINISHED =====\n")


def validate_pipeline():
    """Validate an already saved model."""
    print("\n===== VALIDATION PIPELINE STARTED =====")

    try:
        # Load model
        model = load("models/xgb_model.pkl")
        print("✔ Model loaded successfully")

        # Reload & prepare data
        df = pd.read_csv("data/loan_data.csv")
        df_clean = prepare_data(df)

        # Regenerate test set (same split logic)
        _, X_test, y_test = train(df_clean)

        # Evaluate
        acc, auc, report = evaluate(model, X_test, y_test)
        print("\n===== VALIDATION RESULTS =====")
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print(report)

    except FileNotFoundError:
        print("❌ No trained model found. Run --train first.")

    print("===== VALIDATION PIPELINE FINISHED =====\n")


def main():
    parser = argparse.ArgumentParser(description="Loan Default Prediction Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the XGBoost model")
    parser.add_argument("--validate", action="store_true", help="Validate the saved model")

    args = parser.parse_args()

    if args.train:
        train_pipeline()

    elif args.validate:
        validate_pipeline()

    else:
        print("⚠ No command provided. Default: running training pipeline.")
        train_pipeline()


if __name__ == "__main__":
    main()
