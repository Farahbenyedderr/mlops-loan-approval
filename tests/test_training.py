from src.model import prepare_data, train
import pandas as pd

def test_model_training():
    df = pd.read_csv("data/loan_data.csv")
    df2 = prepare_data(df)
    model, X_test, y_test = train(df2)
    assert model is not None
    assert len(X_test) > 0
