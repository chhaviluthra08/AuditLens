"""Unit tests for core/validation.py."""

import numpy as np
import pandas as pd
import pytest

from core.config import AuditLensConfig
from core.errors import InvalidInputError
from core.validation import validate_and_load


def test_validate_and_load_success(valid_dataframe: pd.DataFrame) -> None:
    """Tests successful validation of valid input DataFrame."""
    audit_data = validate_and_load(
        valid_dataframe,
        group_columns=["region", "credit_band"],
        group_tags={"region": "operational", "credit_band": "audit-only"},
        timestamp_column="timestamp",
    )
    assert audit_data.n_rows == 1200
    assert len(audit_data.score) == 1200
    assert len(audit_data.outcome) == 1200
    assert audit_data.group_columns == ["region", "credit_band"]
    assert audit_data.group_tags["region"] == "operational"
    assert audit_data.group_tags["credit_band"] == "audit-only"
    assert audit_data.timestamp_column == "timestamp"

    # SmallRegion has 30 rows (< 100 threshold), should be in warnings & unreliable
    assert any("SmallRegion" in w.message for w in audit_data.warnings)
    assert ("region", "SmallRegion") in audit_data.unreliable_groups


def test_missing_required_columns(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when required columns are absent."""
    df_no_score = valid_dataframe.drop(columns=["score"])
    with pytest.raises(InvalidInputError, match="Missing required column"):
        validate_and_load(df_no_score)

    df_no_outcome = valid_dataframe.drop(columns=["outcome"])
    with pytest.raises(InvalidInputError, match="Missing required column"):
        validate_and_load(df_no_outcome)


def test_null_values_in_required_columns(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when nulls exist in score or outcome."""
    df_null = valid_dataframe.copy()
    df_null.loc[0, "score"] = np.nan
    with pytest.raises(InvalidInputError, match="missing/null"):
        validate_and_load(df_null)

    df_null2 = valid_dataframe.copy()
    df_null2.loc[10, "outcome"] = np.nan
    with pytest.raises(InvalidInputError, match="missing/null"):
        validate_and_load(df_null2)


def test_insufficient_row_count(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when row count is below minimum required."""
    df_small = valid_dataframe.iloc[:500].copy()
    with pytest.raises(InvalidInputError, match="minimum required is 1000"):
        validate_and_load(df_small)


def test_score_out_of_bounds(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when score values lie outside [0, 1]."""
    df_high = valid_dataframe.copy()
    df_high.loc[0, "score"] = 1.05
    with pytest.raises(InvalidInputError, match=r"outside \[0.0, 1.0\]"):
        validate_and_load(df_high)

    df_low = valid_dataframe.copy()
    df_low.loc[0, "score"] = -0.01
    with pytest.raises(InvalidInputError, match=r"outside \[0.0, 1.0\]"):
        validate_and_load(df_low)


def test_invalid_outcome_values(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when outcome contains non-binary values."""
    df_bad = valid_dataframe.copy()
    df_bad.loc[5, "outcome"] = 2
    with pytest.raises(InvalidInputError, match="invalid classes"):
        validate_and_load(df_bad)


def test_single_class_outcome(valid_dataframe: pd.DataFrame) -> None:
    """Tests failure when only one outcome class exists."""
    df_single = valid_dataframe.copy()
    df_single["outcome"] = 0
    with pytest.raises(InvalidInputError, match="both classes"):
        validate_and_load(df_single)


def test_custom_config_thresholds(valid_dataframe: pd.DataFrame) -> None:
    """Tests that custom configuration thresholds apply correctly."""
    config = AuditLensConfig(min_rows=500, small_group_threshold=20)
    df_500 = valid_dataframe.iloc[:500].copy()
    # 500 rows should now pass because min_rows is 500
    audit_data = validate_and_load(df_500, group_columns=["region"], config=config)
    assert audit_data.n_rows == 500


def test_small_group_flagged_without_error(valid_dataframe: pd.DataFrame) -> None:
    """Proves a group with < small_group_threshold rows is flagged unreliable without error."""
    df = valid_dataframe.copy()
    # Make a tiny group with 15 rows
    df.loc[:14, "tier"] = "RareTier"
    df.loc[15:, "tier"] = "CommonTier"

    # Must succeed without raising an exception
    audit_data = validate_and_load(df, group_columns=["tier"])
    assert ("tier", "RareTier") in audit_data.unreliable_groups
    warning_msgs = [
        w.message for w in audit_data.warnings if w.group_value == "RareTier"
    ]
    assert len(warning_msgs) == 1
    assert "unreliable (small sample)" in warning_msgs[0].lower()


def test_group_at_or_above_threshold_not_flagged(valid_dataframe: pd.DataFrame) -> None:
    """Proves groups with >= small_group_threshold rows are not flagged as unreliable."""
    df = valid_dataframe.copy()
    # Split 1200 rows evenly into 2 groups of 600 (both well above threshold of 100)
    df["balanced_group"] = np.where(np.arange(len(df)) < 600, "GroupA", "GroupB")

    audit_data = validate_and_load(df, group_columns=["balanced_group"])
    assert len(audit_data.unreliable_groups) == 0
    assert len(audit_data.warnings) == 0
