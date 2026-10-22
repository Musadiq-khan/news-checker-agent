from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()
    print(df.shape)