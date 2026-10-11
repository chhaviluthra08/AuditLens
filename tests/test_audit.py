"""Unit and acceptance tests for core/audit.py."""

import os

import pandas as pd

from core.audit import audit_group_column, compute_wilson_ci
from core.splits import create_stratified_split
from core.validation import validate_and_load
from scripts.make_demo_data import generate_demo_datasets


def test_wilson_confidence_interval() -> None:
    """Verifies Wilson score interval computation and properties."""
    low, high = compute_wilson_ci(900, 1000, confidence=0.95)
    assert 0.87 <= low <= 0.90
    assert 0.90 <= high <= 0.92
    assert low < high

    # Edge cases
    assert compute_wilson_ci(0, 0) == (0.0, 1.0)
    low_0, high_0 = compute_wilson_ci(0, 100)
    assert low_0 == 0.0 and high_0 > 0.0


def test_flag_rule_unreliable_never_flagged(valid_dataframe: pd.DataFrame) -> None:
    """Verifies groups marked unreliable (n < threshold) are never flagged, only labelled."""
    df = valid_dataframe.copy()
    # Make SmallRegion have 30 rows (< 100 threshold)
    dataset = validate_and_load(df, group_columns=["region"])
    splits = create_stratified_split(dataset, seed=42)

    res = audit_group_column(dataset, splits, "region")
    small_m = next(m for m in res.group_metrics if m.group_value == "SmallRegion")

    assert small_m.is_unreliable is True
    # Rule requirement: unreliable groups are reported but never flagged
    assert small_m.split_flagged is False
    assert small_m.mondrian_flagged is False
    assert "Unreliable" in small_m.status_label


def test_acceptance_flawed_model_flags_distorted_group() -> None:
    """Acceptance Test: Across 30 seeds, flawed model flags '< 25' on 100% of runs."""
    good_path = "data/demo/good_model.csv"
    flawed_path = "data/demo/flawed_model.csv"
    if not (os.path.exists(good_path) and os.path.exists(flawed_path)):
        generate_demo_datasets()

    flawed_df = pd.read_csv(flawed_path)
    group_cols = ["credit_limit_band", "age_band", "sex", "education", "marital_status"]
    dataset = validate_and_load(flawed_df, group_columns=group_cols)

    detected_count = 0
    for s in range(30):
        splits = create_stratified_split(dataset, seed=s)
        audit_res = audit_group_column(dataset, splits, "age_band")
        if "< 25" in audit_res.flagged_groups:
            detected_count += 1
            m_25 = next(m for m in audit_res.group_metrics if m.group_value == "< 25")
            assert m_25.split_wilson_upper < 0.90
            assert m_25.mondrian_coverage >= 0.87

    assert detected_count == 30, f"Detection failed on {30 - detected_count} seeds"


def test_acceptance_good_model_preserves_coverage() -> None:
    """Acceptance Test: Across 30 seeds and all 5 columns, good model false-flag rate is <= 1/30."""
    good_path = "data/demo/good_model.csv"
    if not os.path.exists(good_path):
        generate_demo_datasets()

    good_df = pd.read_csv(good_path)
    group_cols = ["credit_limit_band", "age_band", "sex", "education", "marital_status"]
    dataset = validate_and_load(good_df, group_columns=group_cols)

    flagged_seeds = 0
    from core.audit import audit_all_groups

    for s in range(30):
        splits = create_stratified_split(dataset, seed=s)
        all_res = audit_all_groups(dataset, splits)
        # '< 25' must never be flagged
        assert "< 25" not in all_res["age_band"].flagged_groups
        if any(len(res.flagged_groups) > 0 for res in all_res.values()):
            flagged_seeds += 1

    # Measured false-flag rate across 30 seeds is 0/30; assert <= 1/30 (<= 3.3%)
    assert flagged_seeds <= 1, f"Good model had {flagged_seeds}/30 flagged seeds"
