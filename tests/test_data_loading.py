import pandas as pd

def test_data_loading():
    df = pd.read_csv("data/loan_data.csv")
    assert df.shape[0] > 0  # il y a des lignes
    assert df.shape[1] > 0  # il y a des colonnes
