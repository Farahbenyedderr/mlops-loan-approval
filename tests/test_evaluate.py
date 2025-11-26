import pandas as pd
from src.model import prepare_data, train, evaluate

def test_evaluate_function():
    # Load and prepare data
    df = pd.read_csv("data/loan_data.csv")
    df_clean = prepare_data(df)

    # Train model
    model, x_test, y_test = train(df_clean)

    # Evaluate
    acc, auc, report = evaluate(model, x_test, y_test)

    # Assertions
    assert acc is not None
    assert isinstance(acc, float)
    assert report is not None
    assert isinstance(report, str)

