"""Split the data into train/test sets and scale the features. Stratified split
preserves the class balance in both sets; the scaler is fit on the training set
only (fitting on the full data would leak test-set information into training).
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


def split_and_scale(dframe, feature_list: list, test_size: float = 0.3,
                     random_state: int = 42):
    """Stratified train/test split followed by standardization (fit on train only).
    Returns X_train, X_test, y_train, y_test, scaler
    """
    X = dframe[feature_list].values
    y = dframe["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler_local = StandardScaler()
    X_tr_scaled = scaler_local.fit_transform(X_tr)
    X_te_scaled = scaler_local.transform(X_te)

    return X_tr_scaled, X_te_scaled, y_tr, y_te, scaler_local


def remove_collinear_features(df_train, feature_list, threshold=0.95):
    """Remove highly correlated features based on a Pearson correlation threshold.

    Computes the correlation matrix on the training set, identifies pairs with
    absolute correlation above the threshold, and drops one feature from each pair
    to reduce multicollinearity before model training.
    """
    corr_matrix = df_train[feature_list].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    collinear_features = [col for col in upper.columns if any(upper[col] > threshold)]
    features_clean = [f for f in feature_list if f not in collinear_features]

    return features_clean, collinear_features

