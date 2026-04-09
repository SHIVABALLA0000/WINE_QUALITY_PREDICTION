import pandas as pd
import numpy as np

def main():
    reference = pd.read_csv("monitoring/reference_data.csv")

    # simulate recent production data
    current = reference.sample(300, random_state=42).reset_index(drop=True)

    # introduce slight drift
    current["alcohol"] += np.random.normal(0.3, 0.1, size=len(current))
    current["sulphates"] += np.random.normal(0.05, 0.02, size=len(current))

    current.to_csv("monitoring/current_data.csv", index=False)
    print("Current data saved: monitoring/current_data.csv")

if __name__ == "__main__":
    main()
