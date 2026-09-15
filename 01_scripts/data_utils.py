"""Reusable data loading utilities, shared across notebooks.

Kept identical to the logic validated in 01_Milestone1.ipynb so every notebook
in this project loads and labels the dataset the same way.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file and return a cleaned dataframe.

    - Drops non-informative identifier columns ('id', and 'Unnamed: 32' if present,
      a stray column found in some public mirrors of this dataset).
    - Encodes 'diagnosis' as a binary label (1 = malignant, 0 = benign).
    """
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["id", "Unnamed: 32"] if c in df.columns])
    df["label"] = (df["diagnosis"] == "M").astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of the 30 continuous morphological feature columns."""
    return [c for c in df.columns if c not in ("diagnosis", "label")]
