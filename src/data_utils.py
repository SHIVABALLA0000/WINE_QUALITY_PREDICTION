import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import RANDOM_STATE

def load_and_prepare_data():
    red = pd.read_csv("data/winequality-red.csv", sep=";")
    white = pd.read_csv("data/winequality-white.csv", sep=";")

    red["wine_type"] = "red"
    white["wine_type"] = "white"

    df = pd.concat([red, white], ignore_index=True)
    df.drop_duplicates(inplace=True,ignore_index=True)
    df.columns = df.columns.str.strip()

    X = df.drop(columns=["quality"])
    y_raw = df["quality"].astype(int)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, le


