"""Loading and quality-checking the dataset, shared across notebooks.

Kept identical to the logic validated in 01_Milestone1.ipynb so every notebook
in this project loads, labels and quality-checks the dataset the same way.
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


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the count of missing values per column."""
    return df.isna().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    """Return the number of fully duplicated rows."""
    return int(df.duplicated().sum())


def detect_outliers_iqr(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Detect outliers per feature using the 1.5*IQR rule.

    Returns a dataframe with the number and percentage of samples flagged as
    outliers for each feature, sorted by outlier count (descending).
    """
    records = []
    for f in feature_list:
        q1, q3 = df[f].quantile(0.25), df[f].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[f] < lower) | (df[f] > upper)).sum()
        records.append({"feature": f, "n_outliers": n_out,
                         "pct_outliers": round(100 * n_out / len(df), 2)})
    return pd.DataFrame(records).sort_values("n_outliers", ascending=False)
