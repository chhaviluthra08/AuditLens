"""Unit tests for core/splits.py."""

import numpy as np
import pandas as pd
import pytest

from core.errors import InvalidInputError
from core.splits import create_stratified_split
from core.validation import validate_and_load


def test_stratified_split_proportions_and_sizes(valid_dataframe: pd.DataFrame) -> None:
    """Verifies that split partitions 50/50 and preserves outcome base rate."""
    dataset = validate_and_load(valid_dataframe)
    splits = create_stratified_split(dataset)

    assert splits.n_cal == 600
    assert splits.n_test == 600
    assert len(splits.cal_indices) == 600
    assert len(splits.test_indices) == 600

    # Ensure sets are completely disjoint
    assert len(set(splits.cal_indices).intersection(set(splits.test_indices))) == 0

    # Stratified balance check
    cal_rate = splits.cal_outcome.mean()
    test_rate = splits.test_outcome.mean()
    assert abs(cal_rate - test_rate) < 0.01


def test_deterministic_seed_reproducibility(valid_dataframe: pd.DataFrame) -> None:
    """Verifies that fixed seeds yield identical splits and differing seeds differ."""
    dataset = validate_and_load(valid_dataframe)
    split_a = create_stratified_split(dataset, seed=42)
    split_b = create_stratified_split(dataset, seed=42)
    split_c = create_stratified_split(dataset, seed=999)

    np.testing.assert_array_equal(split_a.cal_indices, split_b.cal_indices)
    assert not np.array_equal(split_a.cal_indices, split_c.cal_indices)


def test_custom_ratio_split(valid_dataframe: pd.DataFrame) -> None:
    """Tests custom calibration split ratio."""
    dataset = validate_and_load(valid_dataframe)
    splits = create_stratified_split(dataset, cal_ratio=0.70)

    assert abs(splits.n_cal - 840) <= 1
    assert abs(splits.n_test - 360) <= 1
    assert splits.n_cal + splits.n_test == 1200


def test_split_failure_on_too_few_rows() -> None:
    """Verifies InvalidInputError if dataset is too small to split."""
    tiny_df = pd.DataFrame({"score": [0.2], "outcome": [0]})

    # Directly mock dataset object with n_rows = 1
    class DummyDataset:
        n_rows = 1
        data = tiny_df
        outcome = tiny_df["outcome"]

    with pytest.raises(InvalidInputError, match="at least 2 rows"):
        create_stratified_split(DummyDataset())  # type: ignore[arg-type]
