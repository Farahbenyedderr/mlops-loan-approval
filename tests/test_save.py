import os
import pandas as pd
from src.model import prepare_data, train, save

def test_save_model():
    df = pd.read_csv("data/loan_data.csv")
    df_clean = prepare_data(df)

    model, _, _ = train(df_clean)

    save("models/xgb_model.pkl", model)

    assert os.path.exists("models/xgb_model.pkl")

    # clean
    os.remove("models/xgb_model.pkl")


