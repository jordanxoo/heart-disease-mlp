import pandas as pd
import numpy as np

DEFAULT_PATH = "data/heart_disease_uci.csv"

def load_data(source: str = DEFAULT_PATH) -> pd.DataFrame:

    df = pd.read_csv(source)

    df = df.drop(columns=["id", "dataset"])

    for col in ["fbs", "exang"]:
        df[col] = df[col].map({True: 1, False: 0})

    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

  
    df["slope"] = df["slope"].map({
        "upsloping": 1,
        "flat": 2,
        "downsloping": 3,
    })

    df = df.rename(columns={"thalch": "thalach"})

    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])


    df = pd.get_dummies(df, columns=["cp", "restecg", "thal"], dtype=int)

    return df

if __name__ == "__main__":
    df = load_data()
    print(f"Kształt: {df.shape}")
    print(f"\nKolumny:\n{df.columns.tolist()}")
    print(f"\nTypy danych:\n{df.dtypes}")
    print(f"\nBraki danych:\n{df.isnull().sum()}")
    print(f"\nRozkład klas:\n{df['target'].value_counts()}")