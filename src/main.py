import pandas as pd
from model import prepare_data, train, evaluate, save


def prepare_data_main(file_path="data/loan_data.csv"):
    df = pd.read_csv(file_path)
    
    clean_data, training_data, testing_data = prepare_data(df)
    
    clean_data.to_csv("data/cleaned_loan_data.csv", index=False)
    training_data.to_csv("data/train_data.csv", index=False)
    testing_data.to_csv("data/test_data.csv", index=False)

def train_model_main(file_path="data/train_data.csv"):
    train_df = pd.read_csv(file_path)
    
    model = train(train_df)
    joblib.dump(model, "models/trained_model.pkl")


def evaluate_model_main(model_path="models/trained_model.pkl", test_data_path="data/test_data.csv"):
    
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_data_path)
    
    acc, auc, report = evaluate(model,test_df)
    
    print("Model Evaluation:")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)


def main():
    df = pd.read_csv("data/loan_data.csv")
    clean_data, training_data, testing_data = prepare_data(df)
    model = train(training_data)
    acc, auc, report = evaluate(model,testing_data)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)

    save(model)


if __name__ == "__main__":
    main()
