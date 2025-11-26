import argparse
import pandas as pd
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, save_metrics, load_model, load_metrics

def train_pipeline():
    """Execute the complete training pipeline"""
    print("Starting training pipeline...")
    
    # Load data
    df = pd.read_csv("loan_data.csv")
    print("Data loaded successfully")
    
    # Prepare data
    df_clean = prepare_data(df)
    
    # Train the model
    model, X_test, y_test, X_train, y_train = train_model(df_clean)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save the model and metrics
    save_model(model, "models/svc_model.pkl")
    save_metrics(metrics, "models/svc_metrics.json")
    
    print("Pipeline executed successfully!")
    print(f"Model accuracy: {metrics['accuracy']:.2%}")
    print(f"AUC score: {metrics['auc_score']:.3f}")

def validate_pipeline():
    """Validate existing model"""
    print("Validating model...")
    
    try:
        model = load_model("models/svc_model.pkl")
        metrics = load_metrics("models/svc_metrics.json")
        
        print("Model loaded successfully")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"AUC score: {metrics['auc_score']:.3f}")
    except FileNotFoundError:
        print("No trained model found. Run with --train first.")

def main():
    parser = argparse.ArgumentParser(description='Loan Default Prediction Pipeline')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--validate', action='store_true', help='Validate existing model')
    
    args = parser.parse_args()
    
    if args.train:
        train_pipeline()
    elif args.validate:
        validate_pipeline()
    else:
        # Default: run training
        print("No command specified. Running training pipeline...")
        train_pipeline()

if __name__ == "__main__":
    main()