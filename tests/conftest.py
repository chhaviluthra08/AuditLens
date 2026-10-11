"""Shared test fixtures for AuditLens test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Generates a valid 1200-row synthetic credit dataset for testing."""
    rng = np.random.default_rng(42)
    n = 1200
    scores = rng.beta(2, 5, size=n)
    # Outcome correlated with score
    outcomes = (rng.uniform(0, 1, size=n) < scores).astype(int)
    # Ensure both classes exist
    outcomes[0] = 0
    outcomes[1] = 1

    regions = rng.choice(["North", "South", "East", "West"], size=n).astype(object)
    # Include one small group with 30 rows
    regions[-30:] = "SmallRegion"

    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame(
        {
            "score": scores,
            "outcome": outcomes,
            "region": regions,
            "credit_band": rng.choice(["Prime", "Near-Prime", "Subprime"], size=n),
            "timestamp": dates,
        }
    )
