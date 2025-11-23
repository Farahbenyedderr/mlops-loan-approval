import pandas as pd
from model import prepare_data, train, evaluate, save


def main():
    df = pd.read_csv("data/loan_data.csv")

    df_clean = prepare_data(df)

    model, x_test, y_test = train(df_clean)

    acc, auc, report = evaluate(model, x_test, y_test)
    print("Accuracy:", acc)
    print("AUC:", auc)
    print(report)

    save(model)


if __name__ == "__main__":
    main()
