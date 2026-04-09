import pandas as pd
from src.data_utils import load_data   # use the correct function

def main():
    # load full dataset
    X, y, _ = load_data()

    reference = X.copy()
    reference["target"] = y

    reference.to_csv("monitoring/reference_data.csv", index=False)
    print("Reference data saved to monitoring/reference_data.csv")

if __name__ == "__main__":
    main()

