from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_dataset():
    print("Downloading AI4I 2020 dataset from UCI...")

    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features
    targets = dataset.data.targets

    df = pd.concat([features, targets], axis=1)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ai4i2020.csv"
    df.to_csv(output_path, index=False)

    print("\nDownload successful!")
    print(f"Saved to: {output_path}")
    print(f"Dataset shape: {df.shape}")

    print("\nFailure distribution:")
    print(df["Machine failure"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

if __name__ == "__main__":
    download_dataset()
