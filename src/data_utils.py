import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data():
    """
    Load and clean the Wine Quality dataset.

    IMPORTANT:
    - No train/test split here
    - No CV logic here
    - This function ONLY handles data ingestion + basic cleaning

    Returns:
    - X : pd.DataFrame (features)
    - y : np.ndarray (encoded labels)
    - label_encoder : fitted LabelEncoder (artifact)
    """

    # -----------------------------
    # Load raw data
    # -----------------------------
    red = pd.read_csv(r'C:\Users\shiva\Downloads\wine_quality_prediction\data_set\winequality-red.csv', sep=';')
    white= pd.read_csv(r'C:\Users\shiva\Downloads\wine_quality_prediction\data_set\winequality-white.csv', sep=';')
    # Add wine type
    red["wine_type"] = "red"
    white["wine_type"] = "white"

    # Combine datasets
    df = pd.concat([red, white], ignore_index=True)

    # -----------------------------
    # Basic ingestion validation
    # -----------------------------
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.columns = df.columns.str.strip()

    # -----------------------------
    # Split features / target
    # -----------------------------
    X = df.drop(columns=["quality"])
    y_raw = df["quality"].astype(int)

    # -----------------------------
    # Encode target (classification)
    # -----------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder




