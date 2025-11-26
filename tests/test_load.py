import os
import pandas as pd
from src.model import prepare_data, train, save, load

def test_load_model():
    df = pd.read_csv("data/loan_data.csv")
    df_clean = prepare_data(df)

    model, _, _ = train(df_clean)

    save(model, "models/xgb_model.pkl")

    loaded_model = load("models/xgb_model.pkl")

    assert loaded_model is not None

    # clean
    os.remove("models/xgb_model.pkl")
