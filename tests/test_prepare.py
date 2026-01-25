import pandas as pd

from src.model import prepare_data


def test_prepare_outputs_df():
    df = pd.read_csv("data/loan_data.csv")
    df2 = prepare_data(df)
    assert df2 is not None
    assert "loan_status" in df2.columns
    assert df2.isnull().sum().sum() == 0  # pas de valeurs manquantes
