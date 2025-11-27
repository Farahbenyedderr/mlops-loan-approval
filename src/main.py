import argparse
import pandas as pd

from model import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def train_pipeline():
    """Execute the complete training pipeline"""
    print("Starting training pipeline...")

    # Load dataset
    df = pd.read_csv("data/loan_data.csv")
    print("Data loaded successfully")

    # Prepare data
    df_clean = prepare_data(df)

    # Train model
    model, X_test, y_test, feature_names = train_model(df_clean)

    # Evaluate model
    acc, auc, report, cm = evaluate_model(model, X_test, y_test)

    print("Model performance:")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)
    print(cm)

    # Save model
    save_model((model, feature_names), "models/knn_model.pkl")

    print("Pipeline executed successfully!")


def validate_pipeline():
    """Validate a pre-trained model"""
    print("Validating model...")

    try:
        model, feature_names = load_model("models/knn_model.pkl")
        print("Model loaded successfully")

    except FileNotFoundError:
        print("No trained model found. Run with --train first.")


def main():
    parser = argparse.ArgumentParser(description="Loan Default Prediction Pipeline")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--validate", action="store_true", help="Validate the existing model")

    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.validate:
        validate_pipeline()
    else:
        print("No command specified. Running training pipeline...")
        train_pipeline()


if __name__ == "__main__":
    main()
