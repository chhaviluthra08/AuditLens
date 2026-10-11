"""Generates demo datasets: good_model.csv and flawed_model.csv."""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEMO_DIR = "data/demo"
RAW_UCI_PATH = "data/raw/UCI_Credit_Card.csv"
FALLBACK_UCI_PATH = "UCI_Credit_Card.csv"


def _build_synthetic_dataset(n_samples: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic credit dataset when raw UCI file is absent."""
    print("[SYNTHETIC GENERATOR] Generating labelled synthetic credit dataset.")
    rng = np.random.default_rng(seed)

    limit_bands = ["< 50k", "50k-100k", "100k-200k", "200k-300k", "300k+"]
    age_bands = ["< 25", "25-34", "35-49", "50+"]
    sexes = ["Male", "Female"]
    educations = ["Graduate School", "University", "High School", "Other"]
    marriages = ["Married", "Single", "Other"]

    df = pd.DataFrame(
        {
            "credit_limit_band": rng.choice(limit_bands, size=n_samples),
            "age_band": rng.choice(
                age_bands, size=n_samples, p=[0.15, 0.45, 0.30, 0.10]
            ),
            "sex": rng.choice(sexes, size=n_samples),
            "education": rng.choice(educations, size=n_samples),
            "marital_status": rng.choice(marriages, size=n_samples),
        }
    )

    base_risk = rng.beta(2, 6, size=n_samples)
    outcomes = (rng.uniform(0, 1, size=n_samples) < base_risk).astype(int)
    # Ensure both classes exist
    outcomes[0], outcomes[1] = 0, 1

    df["score"] = np.clip(base_risk, 0.001, 0.999)
    df["outcome"] = outcomes
    return df


def _train_uci_model(uci_path: str, seed: int = 42) -> pd.DataFrame:
    """Trains LightGBM model on UCI Default of Credit Card Clients dataset."""
    import lightgbm as lgb

    print(f"[UCI MODEL] Loading raw data from {uci_path}...")
    df = pd.read_csv(uci_path)

    features = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    target_col = "default.payment.next.month"

    X = df[features]
    y = df[target_col]

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_eval)[:, 1]

    eval_df = X_eval.copy()
    eval_df["outcome"] = y_eval.values
    eval_df["score"] = np.clip(scores, 0.001, 0.999)

    eval_df["age_band"] = pd.cut(
        eval_df["AGE"],
        bins=[0, 24, 34, 49, 120],
        labels=["< 25", "25-34", "35-49", "50+"],
    ).astype(str)

    eval_df["credit_limit_band"] = pd.cut(
        eval_df["LIMIT_BAL"],
        bins=[-1, 50000, 100000, 200000, 300000, 1e9],
        labels=["< 50k", "50k-100k", "100k-200k", "200k-300k", "300k+"],
    ).astype(str)

    eval_df["sex"] = (
        eval_df["SEX"].map({1: "Male", 2: "Female"}).fillna("Other").astype(str)
    )
    eval_df["education"] = (
        eval_df["EDUCATION"]
        .map({1: "Graduate School", 2: "University", 3: "High School"})
        .fillna("Other")
        .astype(str)
    )
    eval_df["marital_status"] = (
        eval_df["MARRIAGE"].map({1: "Married", 2: "Single"}).fillna("Other").astype(str)
    )

    cols = [
        "score",
        "outcome",
        "credit_limit_band",
        "age_band",
        "sex",
        "education",
        "marital_status",
    ]
    return eval_df[cols].reset_index(drop=True)


def generate_demo_datasets(seed: int = 42) -> Tuple[str, str]:
    """Generates good_model.csv and flawed_model.csv and saves to data/demo/."""
    os.makedirs(DEMO_DIR, exist_ok=True)

    uci_file = None
    if os.path.exists(RAW_UCI_PATH):
        uci_file = RAW_UCI_PATH
    elif os.path.exists(FALLBACK_UCI_PATH):
        uci_file = FALLBACK_UCI_PATH

    if uci_file:
        good_df = _train_uci_model(uci_file, seed=seed)
    else:
        good_df = _build_synthetic_dataset(seed=seed)

    # Flawed model: Distort '< 25' age band
    flawed_df = good_df.copy()
    rng = np.random.default_rng(seed)

    distorted_group_col = "age_band"
    distorted_group_val = "< 25"
    distorted_mask = flawed_df[distorted_group_col] == distorted_group_val

    distorted_scores = flawed_df["score"].to_numpy().copy()
    # Add subtle background noise to all scores
    noise = rng.normal(0, 0.02, size=len(distorted_scores))
    distorted_scores = np.clip(distorted_scores + noise, 0.001, 0.999)

    # Heavily compress risk for < 25 (overconfident approval)
    distorted_scores[distorted_mask] = np.clip(
        distorted_scores[distorted_mask] * 0.30, 0.001, 0.999
    )
    flawed_df["score"] = distorted_scores

    good_path = os.path.join(DEMO_DIR, "good_model.csv")
    flawed_path = os.path.join(DEMO_DIR, "flawed_model.csv")

    good_df.to_csv(good_path, index=False)
    flawed_df.to_csv(flawed_path, index=False)

    print(f"Generated: {good_path} ({len(good_df)} rows)")
    print(f"Generated: {flawed_path} ({len(flawed_df)} rows)")
    print(
        f"Distorted group for flawed model: {distorted_group_col} == '{distorted_group_val}'"
    )

    return good_path, flawed_path


if __name__ == "__main__":
    generate_demo_datasets()
