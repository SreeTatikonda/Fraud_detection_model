"""
Data loader for the ULB Credit Card Fraud Detection dataset.
284,807 transactions, 492 fraud (0.17%), 30 features (V1-V28 PCA + Time + Amount).

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv in the project root before running train.py.

Usage:
    from fraud_detection.data_loader import load_creditcard
    X_train, X_test, y_train, y_test = load_creditcard()
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_creditcard(
    path: str = "creditcard.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> tuple:
    """
    Loads the ULB credit card dataset and returns train/test splits.

    Features:
        Time    seconds elapsed since first transaction in the dataset
        V1-V28  PCA-transformed original features (confidential)
        Amount  transaction amount in EUR

    Target:
        Class   1 = fraud, 0 = legitimate

    Returns:
        X_train, X_test, y_train, y_test  as numpy arrays
        All arrays are float32 / int32.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at {p.absolute()}\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Place creditcard.csv in the project root."
        )

    print(f"Loading {p} ...")
    df = pd.read_csv(p)

    # Basic validation
    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print(f"  Rows: {len(df):,}   Fraud: {df['Class'].sum():,} "
          f"({df['Class'].mean()*100:.4f}%)")

    # Feature matrix: Time + V1-V28 + Amount = 30 features
    # Pad to 35 dims to match the pipeline's expected feature vector size
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].values.astype(np.float32)

    # Pad with 5 zeros to reach 35 dims (slots for future features)
    padding = np.zeros((len(X), 5), dtype=np.float32)
    X = np.hstack([X, padding])   # (N, 35)

    y = df["Class"].values.astype(np.int32)

    # Stratified split to preserve fraud rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

    print(f"  Train: {len(X_train):,} ({y_train.sum()} fraud)")
    print(f"  Test:  {len(X_test):,}  ({y_test.sum()} fraud)")

    return X_train, X_test, y_train, y_test


def load_for_anomaly(path: str = "creditcard.csv") -> tuple:
    """
    Returns only legitimate transactions for unsupervised training.
    The anomaly detector (IsolationForest + Autoencoder) should be trained
    exclusively on normal sessions so it learns what 'normal' looks like.

    Returns:
        X_legit   (N_legit, 35) float32
        X_all     (N_total, 35) float32   for evaluation
        y_all     (N_total,)    int32
    """
    X_train, X_test, y_train, y_test = load_creditcard(path, scale=True)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    X_legit = X_train[y_train == 0]
    print(f"  Legitimate training samples: {len(X_legit):,}")
    return X_legit, X_all, y_all