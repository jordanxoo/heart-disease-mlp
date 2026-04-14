import pandas as pd
import numpy as np

DEFAULT_PATH = "data/heart_disease_uci.csv"

def load_data(source: str = DEFAULT_PATH) -> pd.DataFrame:
    
    df = pd.read_csv(source)

    df = df.drop(columns=["id", "dataset"])

    for col in ["fbs", "exang"]:
        df[col] = df[col].map({True: 1, False: 0})

    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    df["cp"] = df["cp"].map({
        "typical angina": 1,
        "atypical angina": 2,
        "non-anginal": 3,
        "asymptomatic": 4,
    })

    df["restecg"] = df["restecg"].map({
        "normal": 0,
        "lv hypertrophy": 1,
        "st-t abnormality": 2,
    })

    df["slope"] = df["slope"].map({
        "upsloping": 1,
        "flat": 2,
        "downsloping": 3,
    })

    df["thal"] = df["thal"].map({
        "normal": 1,
        "fixed defect": 2,
        "reversable defect": 3,
    })

    df = df.rename(columns={"thalch": "thalach"})

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Kształt: {df.shape}")
    print(f"\nTypy danych:\n{df.dtypes}")
    print(f"\nBraki danych:\n{df.isnull().sum()}")
    print(f"\nRozkład klas:\n{df['target'].value_counts()}")
