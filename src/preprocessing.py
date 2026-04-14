import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=test_size,random_state=random_state,stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    return X_train_s,X_test_s,y_train,y_test,scaler,imputer
